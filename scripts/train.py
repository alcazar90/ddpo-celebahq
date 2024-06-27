"""Training script for DDPO fine-tuning on google/ddpm-celebahq-256 model."""

import argparse
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from tqdm.auto import tqdm

from ddpo.config import EPS, Task
from ddpo.ddpo import (
    compute_loss,
    evaluation_loop,
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


# Set up progress bars----------------------------------------------------------
def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def master_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPO")

    parser.add_argument(
        "--wandb_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the model (and optimizer) in wandb as an artifact.",
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
        "--gamma",
        type=float,
        default=0.99,
        help="Discounting factor for the rewards. Common values are 0.9 - 0.99.",
    )
    parser.add_argument(
        "--value_lr",
        type=float,
        default=1e-4,
        help="The learning rate for the value network. For now we use a fixed learning rate for the value network.",
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=9e-8,
        help="The initial learning rate. If the initial_lr is equal to the peak_lr, the learning rate is constant",
    )
    parser.add_argument(
        "--peak_lr",
        type=float,
        default=6e-7,
        help="The peak learning rate at the end of the linear warmup phase. If the initial_lr is equal to the peak_lr, the learning rate is constant.",
    )
    parser.add_argument(
        "--warmup_pct",
        type=float,
        default=0.1,
        help="The percentage of training steps to use for learning rate warmup. Common values are 0.1\% - 10\%. (default: 10%)",
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
        "--output_dir",
        type=str,
        default=".",
        help="output directory to save model ckpt",
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
    args.num_batches = int(args.num_samples_per_epoch // args.batch_size)
    if args.initial_lr == args.peak_lr:
        args.warmup_pct = None
    return args


# Initialize the value network--------------------------------------------------
# Initializatoin user in the original PPO repo implementation
# Details for this implementation in:
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueNetwork(nn.Module):
    """
    Value network to estimate the value of a state.
    The state is the denoised image of the diffusion model
    with shape (C, H, W). In this case, the input shape is
    (3, 256, 256) for CelebA-HQ 256x256 images.
    """

    def __init__(self, input_shape=(3, 256, 256)):
        super(ValueNetwork, self).__init__()
        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.network = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x).squeeze(1)


if __name__ == "__main__":
    args = parse_args()

    # Verify paths and directories--------------------------------------------------
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} not found.")

    if args.resume_from_ckpt is not None:
        # Check if the ckpt is available
        if not os.path.exists(args.resume_from_ckpt):
            raise FileNotFoundError(
                f"Checkpoint file {args.resume_from_ckpt} not found."
            )

    # Check if resume_from_ckpt and resume_from_wandb are not both None, throw an error
    if args.resume_from_ckpt is not None and args.resume_from_wandb is not None:
        raise ValueError(
            "You must provide only a single resume mode for load a model checkpoint file; Providing the ckpt path (resume_from_ckpt) or a W&B artifact name for download (resume_from_wandb) to resume training."
        )

    # Create config for logging-----------------------------------------------------
    logging.info(
        "Number batches (`num_samples_per_epoch / batch_size`): %s", args.num_batches
    )

    # Matplotlib settings-----------------------------------------------------------
    plt.rcParams["figure.max_open_warning"] = (
        args.num_eval_samples * args.num_inner_epochs + 1
    )  # or any number greater than 20
    plt.style.use("seaborn-whitegrid")

    # Pytorch settings--------------------------------------------------------------
    # tf32, performance optimization
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize wandb--------------------------------------------------------------
    if args.wandb_logging:
        import wandb

        if args.task == Task.LAION:
            run = wandb.init(
                project="iddpo-aesthetic-ddpm-celebahq256",
                config=vars(args),
                save_code=True,
            )
        elif args.task == Task.UNDER30:
            run = wandb.init(
                project="iddpo-under30-ddpm-celebahq256",
                config=vars(args),
                save_code=True,
            )
        elif args.task == Task.OVER50:
            run = wandb.init(
                project="iddpo-over50-ddpm-celebahq256",
                config=vars(args),
                save_code=True,
            )
        elif args.task == Task.COMPRESSIBILITY:
            run = wandb.init(
                project="iddpo-compressibility-ddpm-celebahq256",
                config=vars(args),
                save_code=True,
            )
        elif args.task == Task.INCOMPRESSIBILITY:
            run = wandb.init(
                project="iddpo-incompressibility-ddpm-celebahq256",
                config=vars(args),
                save_code=True,
            )
        logging.info("Logging to wandb successful, run %s", run.name)

    # Load models-------------------------------------------------------------------
    # Load ddpm_ckpt from hugging face using diffusers library. Only allowed dppm
    # and compatible with DDIMScheduler
    logging.info("Set experiment seed...")
    torch.manual_seed(args.run_seed)

    logging.info("Loading DDPM model and scheduler...")

    image_pipe = DDPMPipeline.from_pretrained(args.ddpm_ckpt).to(args.device)
    scheduler = DDIMScheduler.from_pretrained(args.ddpm_ckpt)
    scheduler.set_timesteps(
        num_inference_steps=args.num_inference_steps, device=args.device
    )

    # Download and initialize the reward model
    if args.task == Task.LAION:
        reward_model = aesthetic_score(
            device=args.device,
        )
    elif args.task == Task.UNDER30:
        reward_model = under30_old(
            threshold=args.threshold,
            punishment=args.punishment,
            device=args.device,
        )
    elif args.task == Task.OVER50:
        reward_model = over50_old(
            threshold=args.threshold,
            punishment=args.punishment,
            device=args.device,
        )
    elif args.task == Task.COMPRESSIBILITY:
        reward_model = jpeg_compressibility(
            device=args.device,
        )
    elif args.task == Task.INCOMPRESSIBILITY:
        reward_model = jpeg_incompressibility(
            device=args.device,
        )

    # Initialize Policy (diffusion) optimizer
    policy_optimizer = torch.optim.AdamW(
        image_pipe.unet.parameters(),
        weight_decay=args.weight_decay,
    )

    # Initialize value network (TODO: requires input shape)
    value_network = ValueNetwork().to(args.device)

    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=args.value_lr,
    )

    # Resume from ckpt--------------------------------------------------------------
    if args.resume_from_ckpt is not None:
        logging.info("Resume training from ckpt path: %s", args.resume_from_ckpt)
        # Add a descripting message to the wandb
        if args.wandb_logging:
            wandb.run.notes = f"Resuming training from ckpt: {args.resume_from_ckpt}"
        ckpt = torch.load(args.resume_from_ckpt, map_location=torch.device(args.device))
        image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
        policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])

    if args.resume_from_wandb is not None:
        logging.info(
            "Resume training from W&B artifact: %s, will be download at path %s",
            args.resume_from_wandb,
            args.output_dir,
        )
        # Add a descripting message to the wandb
        if args.wandb_logging:
            wandb.run.notes = (
                f"Resuming training from W&B artifact: {args.resume_from_wandb}"
            )
        artifact = run.use_artifact(args.resume_from_wandb)
        ckpt_path = artifact.download(args.output_dir)
        ckpt_path = (
            os.path.join(
                args.output_dir, os.path.basename(args.resume_from_wandb)
            ).split(":")[0]
            + "-ckpt.pth"
        )
        logging.info(" -> ckpt loading from: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=torch.device(args.device))
        image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
        policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])

    # Learning Rate Setting---------------------------------------------------------
    # learning rate with linear warmpup and half-cosine cycle annealing
    # implementation based on: https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-D/01_main-chapter-code/appendix-D.ipynb
    total_training_steps = (
        args.num_epochs * args.num_inner_epochs * args.num_batches
    )  # number of time that parameters are update
    global_step = -1  # global lr counter for the run (training experiment)

    # Set warmup_steps to 0, lr_increment to 0, and min_lr to initial_lr if the
    # learning is contsant.
    if args.initial_lr == args.peak_lr:
        warmup_steps = 0
        lr_increment = 0
        min_lr = args.initial_lr
    else:
        warmup_steps = int(args.warmup_pct * total_training_steps)
        warmup_steps = (
            1 if warmup_steps < 1 else warmup_steps
        )  # avoid ZeroDivisionError
        min_lr = 0.1 * args.initial_lr  # possible an hyperparameter...
        lr_increment = (args.peak_lr - args.initial_lr) / warmup_steps

    logging.info(
        "Total training steps: %s | policy warmup steps: %s | policy lr increment: %s | policy min lr: %s | value lr: %s",
        total_training_steps,
        warmup_steps,
        lr_increment,
        min_lr,
        args.value_lr,
    )
    # Training Loop-----------------------------------------------------------------
    track_lrs = []
    mean_rewards = []
    mean_values = []
    epoch_policy_loss = []
    epoch_value_loss = []
    best_reward = -float("inf")  # initialize best reward

    if args.resume_from_ckpt is not None:
        # check if the ckpt has a best reward
        best_reward = ckpt.get("best_reward", -float("inf"))
        logging.info(
            "Loaded best reward from checkpoint: %s. If the best reward is -inf, it indicates an older checkpoint format without 'best_reward' saved.",
            best_reward,
        )
        # if the manual_best_reward is set, overwrite the best reward. Useful for
        # resuming training from a ckpt without the best reward (old version
        # format, e.g. "Task.LAION-youthful-frost-64-ckpt.pth").
        if args.manual_best_reward is not None:
            best_reward = args.manual_best_reward
            logging.info(
                "Overwriting the best reward with the manual value: %s",
                best_reward,
            )

    logging.info("Initializing RL training loop...")
    for epoch in master_bar(range(args.num_epochs)):
        logging.info("Epoch: %s", epoch + 1)
        if args.wandb_logging:
            logging.info("Close all open figures before starting the epoch...")
            plt.close()
        logging.info(
            " -> recollecting #%s samples in %s batches",
            args.num_samples_per_epoch,
            args.num_batches,
        )

        all_step_preds, log_probs, advantages, all_rewards, all_value_estimates = (
            [],
            [],
            [],
            [],
            [],
        )

        # Collect data in batches
        # ----------------------------------------------------------------------
        # collect in batches the trajectory raw and denoised
        # states (T+1, B, C, H, W), each one, and logprobs (T, B)
        for _ in progress_bar(range(args.num_batches)):
            batch_all_step_preds, batch_denoised_all_step_preds, batch_log_probs = (
                sample_from_ddpm_celebahq(
                    args.batch_size,
                    scheduler,
                    image_pipe,
                    args.device,
                )
            )

            # Estimate advantages: trajectory rewards and value estimates
            # ------------------------------------------------------------------
            # compute reward directly in the final sample for debbuging
            batch_final_rewards = reward_model(batch_all_step_preds[-1])

            # compute the discounted rewards-to-go over the entire denoised
            #  trajectory of samples in the batch, a tensor of shape (T, B)
            batch_trajectory_rewards = []

            for t in range(batch_denoised_all_step_preds.shape[0]):
                batch_trajectory_rewards.append(
                    reward_model(batch_denoised_all_step_preds[t])
                )

            # Return the rewards tensor for each trajectory (T, B)
            batch_trajectory_rewards = torch.stack(batch_trajectory_rewards)

            # Compute discounted reward-on-to-go (T, B)
            T, B = batch_trajectory_rewards.shape
            batch_trajectory_discounted_rewards = torch.zeros_like(
                batch_trajectory_rewards
            )
            # Initialize reward-to-go for the last time step
            batch_trajectory_discounted_rewards[-1] = batch_trajectory_rewards[-1]
            # Iterate over each timestep in reverse order to accumulate discounted rewards
            for t in reversed(range(T - 1)):
                batch_trajectory_discounted_rewards[t] = (
                    batch_trajectory_rewards[t]
                    + args.gamma * batch_trajectory_discounted_rewards[t + 1]
                )

            # Estimate the value over the entire denoised trajectory (T, B)
            batch_value_estimates = []
            for t in range(batch_denoised_all_step_preds.shape[0]):
                batch_value_estimates.append(
                    value_network(batch_denoised_all_step_preds[t])
                )

            # Return the value estimates tensor (T, B)
            batch_value_estimates = torch.stack(batch_value_estimates)

            # compute the advantages
            batch_trajectory_advantages = (
                batch_trajectory_discounted_rewards - batch_value_estimates
            )

            # store information...
            # TODO: store rewards and discounted rewards on the future...
            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            advantages.append(batch_trajectory_advantages)
            all_rewards.append(batch_final_rewards)
            all_value_estimates.append(batch_value_estimates)

        # concatenate the collected data
        # ----------------------------------------------------------------------
        all_step_preds = torch.cat(
            all_step_preds, dim=1
        )  # concatenate across batch dim
        log_probs = torch.cat(log_probs, dim=1)  # concatenate across batch dim
        advantages = torch.cat(advantages)
        all_rewards = torch.cat(all_rewards)
        values = torch.cat(all_value_estimates)

        # standardize the advantages across the entire collected data
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # save the mean reward of the current samples
        mean_rewards.append(all_rewards.mean().item())
        logging.info(" -> mean reward: %s", mean_rewards[-1])

        # NEW: save the mean value of the current samples
        mean_values.append(values.mean().item())
        logging.info(" -> mean value: %s", mean_values[-1])

        if args.wandb_logging:
            logging.info("Logging reward statistics and an image batch to wandb...")
            wandb.log(
                {
                    "mean_reward": mean_rewards[-1],
                    "std_reward": all_rewards.std().item(),
                    "min_reward": all_rewards.min().item(),
                    "max_reward": all_rewards.max().item(),
                    "mean_value": mean_values[-1],
                    "std_value": values.std().item(),
                    "reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy()),
                    "img batch": [
                        wandb.Image(
                            Image.fromarray(img),
                            caption=f"{args.task} ({epoch+1}ep): {reward.item()}",
                        )
                        for img, reward in zip(
                            decode_tensor_to_np_img(
                                all_step_preds[-1], melt_batch=False
                            ),
                            all_rewards,
                        )
                    ],
                }
            )

        # clean variables
        logging.info(" -> free GPU memory")
        del batch_all_step_preds
        del batch_denoised_all_step_preds
        del batch_log_probs
        del batch_trajectory_discounted_rewards
        del batch_trajectory_rewards
        del batch_trajectory_advantages
        del batch_final_rewards
        flush()

        # ~~ evaluation step ~~
        # Ensure the devset is initialized with the initial model parameters before any updates.
        if args.eval_every_each_epoch is not None and epoch == 0 and args.wandb_logging:
            logging.info(
                "Initializing devset with images and reward trajectories using seed %s, prior to the first update of model parameters at epoch %s.",
                args.eval_rnd_seed,
                epoch + 1,
            )
            eval_imgs, eval_rdf, _, _ = evaluation_loop(
                reward_model,
                scheduler,
                image_pipe,
                args.device,
                num_samples=args.num_eval_samples,
                random_seed=args.eval_rnd_seed,
            )
            initial_eval_samples = eval_imgs.detach().cpu().clone()
            initial_eval_trajectories = eval_rdf.copy()
        # ~~ end evaluation step ~~

        # Split data into minibatches and update the parameters (exploitation)
        # ----------------------------------------------------------------------
        # For num_inner_epochs times, we go over each sample compute the loss,
        # backpropagate, and update our diffusion model.
        pg_inner_loop_losses = []
        value_inner_loop_losses = []

        logging.info("Starting inner loop...")
        for inner_epoch in progress_bar(range(args.num_inner_epochs)):
            # chunk the samples collected into batches
            # all_step_preds and log_probs (T+1, B, C, H, W)
            # and advantages (T, B) and values (T, B)
            all_step_preds_chunked = torch.chunk(
                all_step_preds, args.num_batches, dim=1
            )
            log_probs_chunked = torch.chunk(log_probs, args.num_batches, dim=1)
            # NOTE: why we chunk the advantages and values across dim=0?
            advantages_chunked = torch.chunk(advantages, args.num_batches, dim=0)
            rewards_chunked = torch.chunk(all_rewards, args.num_batches, dim=0)
            values_chunked = torch.chunk(values, args.num_batches, dim=0)

            pg_loss_value = 0.0
            value_loss_value = 0.0

            # now we start to iterate over the batches (manual dataloader)
            for i in progress_bar(range(len(all_step_preds_chunked))):
                policy_optimizer.zero_grad()
                global_step += 1  # lr counter

                if args.initial_lr == args.peak_lr:
                    # Skip the warmup phase and cosine annealing if the initial_lr
                    # is equal to the peak_lr (fix learning rate)
                    lr = args.initial_lr
                    logging.info(
                        "training step %s / %s, lr constant: %s",
                        global_step,
                        total_training_steps,
                        lr,
                    )
                else:
                    # Adjust the learning rate based on the current phase: warmup/cosine annealing
                    if global_step < warmup_steps:
                        # Linear warmup
                        lr = args.initial_lr + global_step * lr_increment
                        logging.info(
                            "training step %s / %s, lr in warmup phase: %s",
                            global_step,
                            total_training_steps,
                            lr,
                        )
                    else:
                        # Cosine annealing after warmup
                        progress = (global_step - warmup_steps) / (
                            total_training_steps - warmup_steps
                        )
                        lr = min_lr + (args.peak_lr - min_lr) * 0.5 * (
                            1 + math.cos(math.pi * progress)
                        )
                        logging.info(
                            "training step %s / %s, lr in cosine annealing phase: %s",
                            global_step,
                            total_training_steps,
                            lr,
                        )

                # Apply the calculated learning rate to the policy optimizer
                for param_group in policy_optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(policy_optimizer.param_groups[0]["lr"])

                # Obtain the loss value and the ratio of the importance weight
                pg_loss, prob_ratio, pct_clipped_ratios, KL = compute_loss(
                    all_step_preds_chunked[i],
                    log_probs_chunked[i],
                    advantages_chunked[i],
                    args.clip_advantages,
                    args.clip_ratio,
                    image_pipe,
                    scheduler,
                    args.device,
                )  # loss.backward happens inside

                # Compute the value loss using rewards and values
                # minibatches
                mb_value_estimates_flat = values_chunked[i].view(-1)
                mb_rewards_flat = rewards_chunked[i].view(-1)
                value_loss = ((mb_value_estimates_flat - mb_rewards_flat) ** 2).mean()
                # backpropagate the value loss
                # TODO: move within compute_loss() with pg_loss or compute
                # pg_loss outside.
                value_loss.backward()

                # Apply gradient clipping after the warmup phase to avoid expliding gradients
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(
                        image_pipe.unet.parameters(),
                        max_norm=1.0,
                    )  # gradient clipping

                policy_optimizer.step()
                value_optimizer.step()

                pg_loss_value += pg_loss
                value_loss_value += value_loss.item()

                if args.wandb_logging:
                    wandb.log(
                        {
                            "pg_loss": pg_loss,
                            "value_loss": value_loss,
                            "pct_clipped_ratios": pct_clipped_ratios,
                            "prob_ratio": wandb.Histogram(
                                prob_ratio.detach().cpu().numpy(),
                            ),
                            "KL (current vs old policy)": KL,
                            "epoch": epoch,
                            "batch": i,
                            "learning_rate": lr,
                        },
                    )

            pg_inner_loop_losses.append(pg_loss_value / args.num_batches)
            value_inner_loop_losses.append(value_loss_value / args.num_batches)

            if args.wandb_logging:
                wandb.log(
                    {
                        "average policy loss": pg_inner_loop_losses[-1],
                        "inner_epoch": inner_epoch,
                    },
                    {
                        "average value loss": value_inner_loop_losses[-1],
                        "inner_epoch": inner_epoch,
                    },
                )

            logging.info(
                " -> average policy loss: %s | average value loss: %s | inner epoch: %s",
                pg_inner_loop_losses[-1],
                value_inner_loop_losses[-1],
                inner_epoch + 1,
            )

        epoch_policy_loss.append(pg_inner_loop_losses)
        epoch_value_loss.append(value_inner_loop_losses)

        # Start evaluation loop (each args.eval_every_each_epoch)
        # ----------------------------------------------------------------------
        if (
            args.eval_every_each_epoch is not None
            and (((epoch + 1) % args.eval_every_each_epoch) == 0 or epoch == 0)
            and args.wandb_logging
        ):
            logging.info("Evaluating model on epoch %s", epoch + 1)
            eval_imgs, eval_rdf, eval_logp, k = evaluation_loop(
                reward_model,
                scheduler,
                image_pipe,
                args.device,
                num_samples=args.num_eval_samples,
                random_seed=args.eval_rnd_seed,
            )

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
                            decode_tensor_to_np_img(
                                o_img.unsqueeze(0), melt_batch=True
                            ),
                        ),
                    ),
                    wandb.Image(
                        Image.fromarray(
                            decode_tensor_to_np_img(
                                c_img.unsqueeze(0), melt_batch=True
                            ),
                        ),
                    ),
                    eval_rdf[rc][-1:].item(),
                    initial_eval_trajectories[rc][-1:].item(),
                    eval_rdf[rc][-1:].item()
                    - initial_eval_trajectories[rc][-1:].item(),
                    wandb.Image(
                        plt,
                    ),
                )
            wandb.log({"eval_table": table}, commit=False)
            plt.close()
            eval_mean_reward = eval_rdf.iloc[-1, :].mean()
            logging.info(
                " -> eval mean reward (%s epoch): %s", epoch + 1, eval_mean_reward
            )
            wandb.log({"eval_mean_reward": eval_mean_reward})
            del eval_imgs
            del eval_rdf
            del eval_logp
            del k
            flush()

            # save model ckpt if the current mean reward is better than the best reward and save_model is True...
            if eval_mean_reward > best_reward and args.save_model:
                logging.info(
                    " -> saving model ckpt for run %s, current mean reward: %s | best reward: %s",
                    run.name,
                    eval_mean_reward,
                    best_reward,
                )
                # Save unet weights, optimizer state (policy and value), and best_reward.
                ckpt_path = f"{args.output_dir}/{args.task}-{run.name}-ckpt.pth"
                torch.save(
                    {
                        "model_state_dict": image_pipe.unet.state_dict(),
                        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
                        "value_optimizer_state_dict": value_optimizer.state_dict(),
                        "best_reward": eval_mean_reward,
                    },
                    ckpt_path,
                )
                logging.info(" -> ckpt saved at: %s", ckpt_path)
                best_reward = eval_mean_reward
                # Create a new artifact (or overwrite the existing one)
                artifact = wandb.Artifact(f"{args.task}-{run.name}", type="model")
                artifact.add_file(ckpt_path)
                run.log_artifact(artifact)
            logging.info("End evaluation loop")
        # # ~~ end of evaluation ~~

        # clean variables
        logging.info(" -> free GPU memory")
        del all_step_preds_chunked
        del log_probs_chunked
        del advantages_chunked
        del rewards_chunked
        del values_chunked

        flush()

    if args.wandb_logging:
        wandb.finish()
