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
    compute_loss_new_modified,
    compute_loss,
    compute_loss_new_modified_cliped_ratio,
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
from ddpo.sampling import sample_from_ddpm_celebahq, improved_sample_from_ddpm_celebahq
from ddpo.utils import decode_tensor_to_np_img, flush

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# Set up progress bars----------------------------------------------------------
def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def master_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
parser = argparse.ArgumentParser(description="Train DDPO")

parser.add_argument(
    "--wandb_logging",
    type=bool,
    default=True,
)
parser.add_argument(
    "--task",
    type=Task,
    choices=list(Task),
    default=Task.LAION,
    help="The downstream task can be one of the following: aesthetic score, under30 years old, over50 years old, jpeg compressibility, jpeg incompressibility.",
)
parser.add_argument(
    "--num_samples_per_epoch",
    type=int,
    default=10,
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
)
parser.add_argument(
    "--num_inner_epochs",
    type=int,
    default=1,
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=40,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-6,
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--clip_advantages",
    type=float,
    default=2.5,
)
parser.add_argument(
    "--clip_ratio",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--mean_zone_interest_sampling",
    type=int,
    default=30,
)
parser.add_argument(
    "--num_of_segments",
    type=int,
    default=10,
)
parser.add_argument(
    "--ddpm_ckpt",
    type=str,
    default="google/ddpm-celebahq-256",
)
parser.add_argument(
    "--run_seed",
    type=int,
    default=5633313988,
)
parser.add_argument(
    "--eval_every_each_epoch",
    type=int,
    default=None,
)
parser.add_argument(
    "--eval_rnd_seed",
    type=int,
    default=666,
)
parser.add_argument(
    "--num_eval_samples",
    type=int,
    default=2,
)
parser.add_argument(
    "--resume_from_wandb",
    type=str,
    default=None,
    help="Given a W&B artifact ckpt name, download and resume training from it.",
)
parser.add_argument(
    "--resume_from_ckpt",
    type=str,
    default=None,
)
parser.add_argument(
    "--manual_best_reward",
    type=float,
    default=None,
    help="If you want to manually set the best reward. Useful for resuming training from a ckpt without the best reward (old version format).",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
)
parser.add_argument(
    "--output_dir", type=str, default=".", help="output directory to save model ckpt"
)
# threshold and punishment prameter for under30_old and over50_old rewards
parser.add_argument(
    "--threshold",
    type=float,
    default=0.6,
)
parser.add_argument(
    "--punishment",
    type=float,
    default=-1.0,
)

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
mean_zone_interest_sampling = args.mean_zone_interest_sampling
num_of_segments = args.num_of_segments
ddpm_ckpt = args.ddpm_ckpt
resume_from_ckpt = args.resume_from_ckpt
manual_best_reward = args.manual_best_reward
resume_from_wandb = args.resume_from_wandb
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

# Check if resume_from_ckpt and resume_from_wandb are not both None, throw an error
if resume_from_ckpt is not None and resume_from_wandb is not None:
    raise ValueError(
        "You must provide only a single resume mode for load a model checkpoint file; Providing the ckpt path (resume_from_ckpt) or a W&B artifact name for download (resume_from_wandb) to resume training."
    )

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
    "mean_zone_interest_sampling": mean_zone_interest_sampling,
    "number_of_segments": num_of_segments,
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


# Matplotlib settings-----------------------------------------------------------
plt.rcParams["figure.max_open_warning"] = (
    num_eval_samples * num_inner_epochs + 1
)  # or any number greater than 20
plt.style.use("seaborn-whitegrid")

# Pytorch settings--------------------------------------------------------------
# tf32, performance optimization
torch.backends.cuda.matmul.allow_tf32 = True

# Initialize wandb--------------------------------------------------------------
if wandb_logging:
    if task == Task.LAION:
        run = wandb.init(
            project="ddpo-aesthetic-ddpm-celebahq256",
            config=config,
        )
    elif task == Task.UNDER30:
        run = wandb.init(
            project="ddpo-under30-ddpm-celebahq256",
            config=config,
        )
    elif task == Task.OVER50:
        run = wandb.init(
            project="ddpo-over50-ddpm-celebahq256",
            config=config,
        )
    elif task == Task.COMPRESSIBILITY:
        run = wandb.init(
            project="ddpo-compressibility-ddpm-celebahq256",
            config=config,
        )
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
    reward_model = aesthetic_score(
        device=device,
    )
elif task == Task.UNDER30:
    reward_model = under30_old(
        threshold=threshold,
        punishment=punishment,
        device=device,
    )
elif task == Task.OVER50:
    reward_model = over50_old(
        threshold=threshold,
        punishment=punishment,
        device=device,
    )
elif task == Task.COMPRESSIBILITY:
    reward_model = jpeg_compressibility(
        device=device,
    )
elif task == Task.INCOMPRESSIBILITY:
    reward_model = jpeg_incompressibility(
        device=device,
    )

# Optimizer
optimizer = torch.optim.AdamW(
    image_pipe.unet.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)  # optimizer

# Resume from ckpt--------------------------------------------------------------
if resume_from_ckpt is not None:
    logging.info("Resume training from ckpt path: %s", resume_from_ckpt)
    # Add a descripting message to the wandb
    if wandb_logging:
        wandb.run.notes = f"Resuming training from ckpt: {resume_from_ckpt}"
    ckpt = torch.load(resume_from_ckpt, map_location=torch.device(device))
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

if resume_from_wandb is not None:
    logging.info(
        "Resume training from W&B artifact: %s, will be download at path %s",
        resume_from_wandb,
        output_dir,
    )
    # Add a descripting message to the wandb
    if wandb_logging:
        wandb.run.notes = f"Resuming training from W&B artifact: {resume_from_wandb}"
    artifact = run.use_artifact(resume_from_wandb)
    ckpt_path = artifact.download(output_dir)
    ckpt_path = (
        os.path.join(output_dir, os.path.basename(resume_from_wandb)).split(":")[0]
        + "-ckpt.pth"
    )
    logging.info(" -> ckpt loading from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
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
    if wandb_logging:
        logging.info("Close all open figures before starting the epoch...")
        plt.close()
    logging.info(
        " -> recollecting #%s samples in %s batches",
        num_samples_per_epoch,
        num_batches,
    )

    all_step_preds, log_probs, advantages, all_rewards, all_initiate_train_steps = [], [], [], [], []

    # sampling `num_samples_per_epoch` images, intermediate states, and
    # final_rewards by collecting each samples_per_batch, until complete
    # batch_size times `num_samples_per_epoch = num_batches * batch_size`
    for _ in progress_bar(range(num_batches)):
        batch_all_step_preds, batch_log_probs, initiate_train_step = improved_sample_from_ddpm_celebahq(
            batch_size,
            scheduler,
            image_pipe,
            device,
            epoch,
            num_epochs,
            mean_zone_interest_sampling,
            num_of_segments)

        # compute reward on the final step (sample), and obtain advantages
        batch_rewards = reward_model(batch_all_step_preds[-1])
        batch_advantages = standardize(batch_rewards)

        # store information...
        all_step_preds.append(batch_all_step_preds)
        log_probs.append(batch_log_probs)
        advantages.append(batch_advantages)
        all_rewards.append(batch_rewards)
        all_initiate_train_steps.append(initiate_train_step)

    all_step_preds = torch.cat(all_step_preds, dim=1)  # concatenate across batch dim
    log_probs = torch.cat(log_probs, dim=1)  # concatenate across batch dim
    advantages = torch.cat(advantages)
    all_rewards = torch.cat(all_rewards)
    logging.info(" Batch intiate train steps for ddpo: %s", all_initiate_train_steps)
    # all_initiate_train_steps = torch.cat(all_initiate_train_steps)

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
    del initiate_train_step
    flush()

    # For num_inner_epochs times, we go over each sample compute the loss,
    # backpropagate, and update our diffusion model.
    inner_loop_losses = []

    logging.info("Starting inner loop...")
    for inner_epoch in progress_bar(range(num_inner_epochs)):
        # chunk them into batches
        all_step_preds_chunked = torch.chunk(all_step_preds, num_batches, dim=1)
        print("Length of all_step_preds_chunked: ", len(all_step_preds_chunked))
        log_probs_chunked = torch.chunk(log_probs, num_batches, dim=1)
        advantages_chunked = torch.chunk(advantages, num_batches, dim=0)
        # all_initiate_train_steps_chunked = torch.chunk(all_initiate_train_steps, num_batches, dim=0)

        loss_value = 0.0
        for i in progress_bar(range(len(all_step_preds_chunked))):
            optimizer.zero_grad()

            # Obtain the loss value and the ratio of the importance weight
            loss, prob_ratio, pct_clipped_ratios, KL = compute_loss_new_modified_cliped_ratio(
                all_step_preds_chunked[i],
                log_probs_chunked[i],
                advantages_chunked[i],
                all_initiate_train_steps[i],
                clip_advantages,
                clip_ratio,
                image_pipe,
                scheduler,
                device,
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

    # if wandb_logging:
    #     wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
    #     wandb.log({"mean_reward": mean_rewards[-1]})

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
                "diff_reward",
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
                eval_rdf[rc][-1:].item() - initial_eval_trajectories[rc][-1:].item(),
                wandb.Image(
                    plt,
                ),
            )
        wandb.log({"eval_table": table}, commit=False)
        plt.close()
        eval_mean_reward = eval_rdf.iloc[-1, :].mean()
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
            ckpt_path = f"{output_dir}/{task}-{run.name}-ckpt.pth"
            torch.save(
                {
                    "model_state_dict": image_pipe.unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_reward": eval_mean_reward,
                },
                ckpt_path,
            )
            logging.info(" -> ckpt saved at: %s", ckpt_path)
            best_reward = eval_mean_reward
            # Create a new artifact (or overwrite the existing one)
            artifact = wandb.Artifact(f"{task}-{run.name}", type="model")
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
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
