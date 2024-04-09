"""Training script for DDPO fine-tuning on google/ddpm-celebahq-256 model."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import torch
import wandb
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from tqdm.auto import tqdm

from ddpo.config import Task
from ddpo.ddpo import (
    compute_loss,
    evaluation_loop,
    standardize,
)
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)
from ddpo.sampling import sample_from_ddpm_celebahq
from ddpo.utils import decode_tensor_to_np_img, flush

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Matplotlib settings-----------------------------------------------------------
plt.rcParams["figure.max_open_warning"] = 60  # or any number greater than 20
plt.style.use("seaborn-whitegrid")


# Set up progress bars----------------------------------------------------------
def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def master_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
parser = argparse.ArgumentParser(description="DDPO")

parser.add_argument("--wandb_logging", type=bool, default=True)
parser.add_argument("--task", type=Task, choices=list(Task), default=Task.LAION)
parser.add_argument("--num_samples_per_epoch", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--num_inner_epochs", type=int, default=1)
parser.add_argument("--num_inference_steps", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--clip_advantages", type=float, default=2.5)
parser.add_argument("--clip_ratio", type=float, default=1e-4)
parser.add_argument("--ddpm_ckpt", type=str, default="google/ddpm-celebahq-256")
parser.add_argument("--run_seed", type=int, default=5633313988)
parser.add_argument("--eval_every_each_epoch", type=int, default=None)
parser.add_argument("--eval_rnd_seed", type=int, default=666)
parser.add_argument("--num_eval_samples", type=int, default=2)
parser.add_argument("--resume_from_ckpt", type=str, default=None)
parser.add_argument(
    "--manual_best_reward",
    type=float,
    default=None,
    help="If you want to manually set the best reward. Useful for resuming training from a ckpt without the best reward (old version format).",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="..",
    help="output directory to save model ckpt",
)

# threshold and punishment prameter for under30_old and over50_old rewards
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--punishment", type=float, default=-1.0)

args = parser.parse_args()

wandb_logging = args.wandb_logging
task = args.task
num_samples_per_epoch = args.num_samples_per_epoch
num_epochs = args.num_epochs
num_inner_epochs = args.num_inner_epochs
num_inference_steps = args.num_inference_steps
batch_size = args.batch_size
lr = args.lr
weight_decay = args.weight_decay
clip_advantages = args.clip_advantages
clip_ratio = args.clip_ratio
ddpm_ckpt = args.ddpm_ckpt
resume_from_ckpt = args.resume_from_ckpt
manual_best_reward = args.manual_best_reward
device = args.device
threshold = args.threshold
punishment = args.punishment
num_batches = num_samples_per_epoch // batch_size
run_seed = args.run_seed
eval_every_each_epoch = args.eval_every_each_epoch
eval_random_seed = args.eval_rnd_seed
num_eval_samples = args.num_eval_samples
output_dir = args.output_dir

# Verify paths and directories--------------------------------------------------
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"Output directory {output_dir} not found.")

if resume_from_ckpt is not None:
    # Check if the ckpt is available
    if not os.path.exists(resume_from_ckpt):
        raise FileNotFoundError(f"Checkpoint file {resume_from_ckpt} not found.")

# Create config for logging-----------------------------------------------------
config = {
    "task": task,
    "num_samples_per_epoch": num_samples_per_epoch,
    "num_epochs": num_epochs,
    "num_inner_epochs": num_inner_epochs,
    "batch_size": batch_size,
    "num_batches": num_batches,
    "num_inference_steps": num_inference_steps,
    "lr": lr,
    "weight_decay": weight_decay,
    "clip_advantages": clip_advantages,
    "clip_ratio": clip_ratio,
    "ddpm_ckpt": ddpm_ckpt,
    "resume_from_ckpt": resume_from_ckpt,
    "manual_best_reward": manual_best_reward,
    "run_seed": run_seed,
    "eval_every_each_epoch": eval_every_each_epoch,
    "eval_rnd_seed": eval_random_seed,
    "num_eval_samples": num_eval_samples,
}

if task in (Task.UNDER30, Task.OVER50):
    config["threshold"] = threshold
    config["punishment"] = punishment

logging.info("Number batches (`num_samples_per_epoch / batch_size`): %s", num_batches)


# Initialize wandb--------------------------------------------------------------
if wandb_logging:
    if task == Task.LAION:
        run = wandb.init(project="ddpo-aesthetic-ddpm-celebahq256", config=config)
    elif task == Task.UNDER30:
        run = wandb.init(project="ddpo-under30-ddpm-celebahq256", config=config)
    elif task == Task.OVER50:
        run = wandb.init(project="ddpo-over50-ddpm-celebahq256", config=config)
    elif task == Task.COMPRESSIBILITY:
        run = wandb.init(project="ddpo-compressibility-ddpm-celebahq256", config=config)
    elif task == Task.INCOMPRESSIBILITY:
        run = wandb.init(
            project="ddpo-incompressibility-ddpm-celebahq256",
            config=config,
        )
    logging.info("Logging to wandb successful, run %s", run.name)


# Load models-------------------------------------------------------------------
# Load ddpm_ckpt from hugging face using diffusers library. Only allowed dppm
# and compatible with DDIMScheduler
logging.info("Set experiment seed...")
torch.manual_seed(run_seed)

logging.info("Loading DDPM model and scheduler...")

image_pipe = DDPMPipeline.from_pretrained(ddpm_ckpt).to(device)
scheduler = DDIMScheduler.from_pretrained(ddpm_ckpt)
scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

# Download and initialize the reward model
if task == Task.LAION:
    reward_model = aesthetic_score()
elif task == Task.UNDER30:
    reward_model = under30_old(threshold=threshold, punishment=punishment)
elif task == Task.OVER50:
    reward_model = over50_old(threshold=threshold, punishment=punishment)
elif task == Task.COMPRESSIBILITY:
    reward_model = jpeg_compressibility()
elif task == Task.INCOMPRESSIBILITY:
    reward_model = jpeg_incompressibility()

# Optimizer
optimizer = torch.optim.AdamW(
    image_pipe.unet.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)  # optimizer

# Resume from ckpt--------------------------------------------------------------
if resume_from_ckpt is not None:
    # Add a descripting message to the wandb
    if wandb_logging:
        wandb.run.notes = f"Resuming training from ckpt: {resume_from_ckpt}"
        wandb.run.save()
        # TODO: Before loading the ckpt, obtain the eval samples' trajectories
        # from the original model and their corresponding reward metric
        # eval_imgs, eval_rdf, eval_logp, k = evaluation_loop(
        #     reward_model,
        #     scheduler,
        #     image_pipe,
        #     device,
        #     num_samples=num_eval_samples,
        #     random_seed=eval_random_seed,
        # )
    ckpt = torch.load(resume_from_ckpt)
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

# Training Loop-----------------------------------------------------------------
logging.info("Initializing RL training loop...")

mean_rewards = []
epoch_loss = []
best_reward = -float("inf")  # initialize best reward

if resume_from_ckpt is not None:
    # check if the ckpt has a best reward
    best_reward = ckpt.get("best_reward", -float("inf"))
    logging.info(
        "Loaded best reward from checkpoint: %s. If the best reward is -inf, it indicates an older checkpoint format without 'best_reward' saved.",
        best_reward,
    )
    # if the manual_best_reward is set, overwrite the best reward. Useful for
    # resuming training from a ckpt without the best reward (old version
    # format, e.g. "Task.LAION-youthful-frost-64-ckpt.pth").
    if manual_best_reward is not None:
        best_reward = manual_best_reward
        logging.info(
            "Overwriting the best reward with the manual value: %s",
            best_reward,
        )

for epoch in master_bar(range(num_epochs)):
    logging.info("Epoch: %s", epoch + 1)
    logging.info(
        " -> recollecting #%s samples in %s batches",
        num_samples_per_epoch,
        num_batches,
    )

    all_step_preds, log_probs, advantages, all_rewards = [], [], [], []

    # sampling `num_samples_per_epoch` images, intermediate states, and
    # final_rewards by collecting each samples_per_batch, until complete
    # batch_size times `num_samples_per_epoch = num_batches * batch_size`
    for _ in progress_bar(range(num_batches)):
        batch_all_step_preds, batch_log_probs = sample_from_ddpm_celebahq(
            batch_size,
            scheduler,
            image_pipe,
            device,
        )

        # compute reward on the final step (sample), and obtain advantages
        batch_rewards = reward_model(batch_all_step_preds[-1])
        batch_advantages = standardize(batch_rewards)

        # store information...
        all_step_preds.append(batch_all_step_preds)
        log_probs.append(batch_log_probs)
        advantages.append(batch_advantages)
        all_rewards.append(batch_rewards)

    all_step_preds = torch.cat(all_step_preds, dim=1)  # concatenate across batch dim
    log_probs = torch.cat(log_probs, dim=1)  # concatenate across batch dim
    advantages = torch.cat(advantages)
    all_rewards = torch.cat(all_rewards)

    # save the mean reward of the current samples
    mean_rewards.append(all_rewards.mean().item())
    logging.info(" -> mean reward: %s", mean_rewards[-1])

    if wandb_logging:
        logging.info("Logging reward statistics and an image batch to wandb...")
        wandb.log({"mean_reward": mean_rewards[-1]})
        wandb.log({"std_reward": all_rewards.std().item()})
        wandb.log({"min_reward": all_rewards.min().item()})
        wandb.log({"max_reward": all_rewards.max().item()})
        wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
        # Nota: sobre las imagenes si suben con memoria ram, no sé si aporta
        # mucho guardar todas
        wandb.log(
            {
                "img batch": [
                    wandb.Image(
                        Image.fromarray(img),
                        caption=f"{task} ({epoch+1}ep): {reward.item()}",
                    )
                    for img, reward in zip(
                        decode_tensor_to_np_img(all_step_preds[-1], melt_batch=False),
                        all_rewards,
                    )
                ],
            },
        )

    # clean variables
    logging.info(" -> free GPU memory")
    del batch_all_step_preds
    del batch_log_probs
    del batch_rewards
    del batch_advantages
    flush()

    # For num_inner_epochs times, we go over each sample compute the loss,
    # backpropagate, and update our diffusion model.
    inner_loop_losses = []

    logging.info("Starting inner loop...")
    for inner_epoch in progress_bar(range(num_inner_epochs)):
        # chunk them into batches
        all_step_preds_chunked = torch.chunk(all_step_preds, num_batches, dim=1)
        log_probs_chunked = torch.chunk(log_probs, num_batches, dim=1)
        advantages_chunked = torch.chunk(advantages, num_batches, dim=0)

        loss_value = 0.0
        for i in progress_bar(range(len(all_step_preds_chunked))):
            optimizer.zero_grad()

            # Obtain the loss value and the ratio of the importance weight
            loss, prob_ratio, pct_clipped_ratios, KL = compute_loss(
                all_step_preds_chunked[i],
                log_probs_chunked[i],
                advantages_chunked[i],
                clip_advantages,
                clip_ratio,
                image_pipe,
                scheduler,
                "cuda",
            )  # loss.backward happens inside

            torch.nn.utils.clip_grad_norm_(
                image_pipe.unet.parameters(),
                1.0,
            )  # gradient clipping
            optimizer.step()
            loss_value += loss

            if wandb_logging:
                wandb.log(
                    {
                        "loss": loss,
                        "pct_clipped_ratios": pct_clipped_ratios,
                        "prob_ratio": wandb.Histogram(
                            prob_ratio.detach().cpu().numpy(),
                        ),
                        "KL (current vs old policy)": KL,
                        "epoch": epoch,
                        "batch": i,
                    },
                )

        inner_loop_losses.append(loss_value / num_batches)
        if wandb_logging:
            wandb.log(
                {"average loss": inner_loop_losses[-1], "inner_epoch": inner_epoch},
            )

        logging.info(
            " -> average loss: %s | inner epoch: %s",
            inner_loop_losses[-1],
            inner_epoch + 1,
        )

    if wandb_logging:
        wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
        wandb.log({"mean_reward": mean_rewards[-1]})

    epoch_loss.append(inner_loop_losses)

    # evaluation loop each X epochs, and at the start and end of training
    # TODO: encapsulate this in a function to make the code more readable.
    # Add an option for create a table in resume ckpt mode to compare the
    # initial, ckpt, and current reward trajectories as well as images.
    if (
        eval_every_each_epoch is not None
        and (((epoch + 1) % eval_every_each_epoch) == 0 or epoch == 0)
        and wandb_logging
    ):
        logging.info("Evaluating model on epoch %s", epoch + 1)
        eval_imgs, eval_rdf, eval_logp, k = evaluation_loop(
            reward_model,
            scheduler,
            image_pipe,
            device,
            num_samples=num_eval_samples,
            random_seed=eval_random_seed,
        )

        # save the initial evaluation samples images (for comparison)
        if epoch == 0:
            initial_eval_samples = eval_imgs.detach().cpu().clone()
            initial_eval_trajectories = eval_rdf.copy()

        # log the evaluation results in a wandb.Table
        table = wandb.Table(
            columns=[
                "original_samples",
                "current_samples",
                "current_final_reward",
                "original_final_reward",
                "reward_trajectory",
            ],
        )

        for (
            o_img,
            c_img,
            rc,
        ) in zip(initial_eval_samples, eval_imgs, eval_rdf):
            # create reward plot trajectory
            plt.figure(figsize=(10, 4))
            plt.plot(
                eval_rdf[rc],
                color="mediumseagreen",
                label="current rwd trajectory",
            )
            plt.plot(
                initial_eval_trajectories[rc],
                color="indianred",
                label="initial rwd trajectory",
            )
            plt.xlim(0, 40)
            plt.grid(color="lightgrey", linewidth=0.4)
            plt.legend(frameon=False)
            table.add_data(
                wandb.Image(
                    Image.fromarray(
                        decode_tensor_to_np_img(o_img.unsqueeze(0), melt_batch=True),
                    ),
                ),
                wandb.Image(
                    Image.fromarray(
                        decode_tensor_to_np_img(c_img.unsqueeze(0), melt_batch=True),
                    ),
                ),
                eval_rdf[rc][-1:].item(),
                initial_eval_trajectories[rc][-1:].item(),
                wandb.Image(
                    plt,
                ),
            )
        wandb.log({"eval_table": table}, commit=False)
        plt.close()
        eval_mean_reward = eval_rdf.iloc[-1, :].mean().item()
        logging.info(" -> eval mean reward (%s epoch): %s", epoch + 1, eval_mean_reward)
        wandb.log({"eval_mean_reward": eval_mean_reward})
        del eval_imgs
        del eval_rdf
        del eval_logp
        del k
        flush()

        # save model ckpt if the current mean reward is better than the best reward
        if eval_mean_reward > best_reward:
            logging.info(
                " -> saving model ckpt for run %s, current mean reward: %s | best reward: %s",
                run.name,
                eval_mean_reward,
                best_reward,
            )
            # Save unet weights, optimizer state, and best_reward.
            torch.save(
                {
                    "model_state_dict": image_pipe.unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_reward": eval_mean_reward,
                },
                f"{output_dir}/{task}-{run.name}-ckpt.pth",
            )
            best_reward = eval_mean_reward
            logging.info("End evaluation loop")
    # # ~~ end of evaluation ~~

    # clean variables
    logging.info(" -> free GPU memory")
    del all_step_preds_chunked
    del log_probs_chunked
    del advantages_chunked

    flush()

if wandb_logging:
    wandb.finish()
