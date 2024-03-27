import logging
import argparse
import wandb

import torch

from diffusers import DDIMScheduler, DDPMPipeline

from PIL import Image
from tqdm import tqdm
# from fastprogress import progress_bar, master_bar

from ddpo.config import Task
from ddpo.utils import flush, decode_tensor_to_np_img
from ddpo.rewards import aesthetic_score, under30_old, over50_old
from ddpo.ddpo import (compute_loss,
                       sample_from_ddpm_celebahq, 
                       standardize)


# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Set up progress bars---------------------------------------------------------
def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)

def master_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
parser = argparse.ArgumentParser(description="DDPO")

parser.add_argument('--wandb_logging', type=bool, default=True)
parser.add_argument('--task', type=Task, choices=list(Task), default=Task.LAION)
parser.add_argument('--num_samples_per_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--num_inner_epochs', type=int, default=1)
parser.add_argument('--num_inference_steps', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip_advantages', type=float, default=2.5)
parser.add_argument('--clip_ratio', type=float, default=1e-4)
parser.add_argument('--ddpm_ckpt', type=str, default="google/ddpm-celebahq-256")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--output_dir", type=str, default="..", help="output directory to save model ckpt")

# threshold and punishment prameter for under30_old and over50_old rewards
parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--punishment', type=float, default=-1.0)

args = parser.parse_args()

wandb_logging = args.wandb_logging
task = args.task
num_samples_per_epoch = args.num_samples_per_epoch
num_epochs = args.num_epochs
num_inner_epochs = args.num_inner_epochs
num_inference_steps = args.num_inference_steps
batch_size = args.batch_size
lr = args.lr
weight_decay= args.weight_decay
clip_advantages = args.clip_advantages
clip_ratio = args.clip_ratio
ddpm_ckpt = args.ddpm_ckpt
device = args.device
threshold = args.threshold
punishment = args.punishment
num_batches = num_samples_per_epoch // batch_size

# Create config for logging-----------------------------------------------------
config = dict(
    task=task,
    num_samples_per_epoch=num_samples_per_epoch,
    num_epochs=num_epochs,
    num_inner_epochs=num_inner_epochs,
    batch_size=batch_size,
    num_batches=num_batches,
    num_inference_steps=num_inference_steps,
    lr=lr,
    weight_decay=weight_decay,
    clip_advantages=clip_advantages,
    clip_ratio=clip_ratio,
    ddpm_ckpt=ddpm_ckpt,
)

if task in (Task.UNDER30, Task.OVER50):
    config["threshold"] = threshold
    config["punishment"] = punishment

logging.info(f"Number batches (`num_samples_per_epoch / batch_size`): {num_batches}")


# Initialize wandb--------------------------------------------------------------
if wandb_logging:
  if task == Task.LAION:
    wandb.init(project="ddpo-aesthetic-ddpm-celebahq256", 
                config=config)
  elif task == Task.UNDER30:
    wandb.init(project="ddpo-under30-ddpm-celebahq256", 
                config=config)
  elif task == Task.OVER50:
    wandb.init(project="ddpo-over50-ddpm-celebahq256", 
                config=config)
  elif task == Task.COMPRESSIBILITY:
    wandb.init(project="ddpo-compressibility-ddpm-celebahq256", 
                config=config)
  elif task == Task.INCOMPRESSIBILITY:
    wandb.init(project="ddpo-incompressibility-ddpm-celebahq256", 
                config=config)
  logging.info("Logging to wandb successful")


# Load models-------------------------------------------------------------------
# Load ddpm_ckpt from hugging face using diffusers library. Only allowed dppm and compatible
# with DDIMScheduler
logging.info("Loading DDPM model and scheduler...")
logging.info("Allowing TF32 for faster matrix multiplication")
torch.backends.cuda.matmul.allow_tf32 = True

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

# Optimizer
optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr, weight_decay=weight_decay) # optimizer



# Training Loop-----------------------------------------------------------------
logging.info("Initializing RL training loop...")

mean_rewards = []
epoch_loss = []

for epoch in master_bar(range(num_epochs)):
    logging.info(f"Epoch: {epoch + 1}")
    logging.info(f" -> recollecting #{num_samples_per_epoch} samples in {num_batches} batches")

    all_step_preds, log_probs, advantages, all_rewards = [], [], [], []

    # sampling `num_samples_per_epoch` images, intermediate states, and final_rewards
    # by collecting each samples_per_batch, until complete batch_size times
    # `num_samples_per_epoch = num_batches * batch_size`
    for _ in progress_bar(range(num_batches)):
      batch_all_step_preds, batch_log_probs = sample_from_ddpm_celebahq(
                                                                batch_size,
                                                                scheduler,
                                                                image_pipe,
                                                                device)

      # compute reward, and obtain advantages by standardizing rewards
      batch_rewards = reward_model(batch_all_step_preds[-1])
      batch_advantages = standardize(batch_rewards)

      # store information...
      all_step_preds.append(batch_all_step_preds)
      log_probs.append(batch_log_probs)
      advantages.append(batch_advantages)
      all_rewards.append(batch_rewards)

    all_step_preds = torch.cat(all_step_preds, dim=1) # concatenate across batch dim
    log_probs = torch.cat(log_probs, dim=1) # concatenate across batch dim
    advantages = torch.cat(advantages)
    all_rewards = torch.cat(all_rewards)


    # save the mean reward of the current samples
    mean_rewards.append(all_rewards.mean().item())
    logging.info(f" -> mean reward: {mean_rewards[-1]}")

    if wandb_logging:
        logging.info("Logging reward statistics and an image batch to wandb...")
        wandb.log({"mean_reward": mean_rewards[-1]})
        wandb.log({"std_reward": all_rewards.std().item()})
        wandb.log({"min_reward": all_rewards.min().item()})
        wandb.log({"max_reward": all_rewards.max().item()})
        wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
        # Nota: sobre las imagenes si suben con memoria ram, no sé si aporta mucho guardar tantasa
        wandb.log(
            {
                "img batch": [
                    wandb.Image(Image.fromarray(img), caption=f"{task} ({epoch+1}ep): {reward.item()}")
                    for img, reward in
                    zip(decode_tensor_to_np_img(all_step_preds[-1], melt_batch=False), all_rewards)
                ]
            }
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

            loss = compute_loss(all_step_preds_chunked[i], log_probs_chunked[i],
                                advantages_chunked[i], clip_advantages, clip_ratio,
                                image_pipe, scheduler, 'cuda'
                                ) # loss.backward happens inside

            torch.nn.utils.clip_grad_norm_(image_pipe.unet.parameters(), 1.0) # gradient clipping
            optimizer.step()
            loss_value += loss

            if wandb_logging:
                wandb.log(
                    {
                        "loss": loss,
                        "epoch": epoch,
                        "batch": i,
                    }
                )

        inner_loop_losses.append(loss_value / num_batches)
        if wandb_logging:
            wandb.log({"average loss": inner_loop_losses[-1],
                       "inner_epoch": inner_epoch})

        logging.info(f" -> average loss: {inner_loop_losses[-1]} | inner epoch: {inner_epoch + 1}")

    if wandb_logging:
      wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
      wandb.log({"mean_reward": mean_rewards[-1]})

    epoch_loss.append(inner_loop_losses)
    # clean variables
    logging.info(" -> free GPU memory")
    del all_step_preds_chunked
    del log_probs_chunked
    del advantages_chunked
    flush()

if wandb_logging:
    wandb.finish()

# save the model
image_pipe.save_pretrained(args.output_dir)
