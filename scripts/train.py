"""Training script for DDPO fine-tuning on google/ddpm-celebahq-256 model."""

import argparse
import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from tqdm.auto import tqdm

from ddpo.config import EPS, Task
from ddpo.ddpo import (
    compute_discounted_returns,
    evaluation_loop,
)
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)
from ddpo.sampling import calculate_log_probs, sample_from_ddpm_celebahq
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
# TODO: Complete helps for each hyperparameter
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
        "--norm-adv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggles advantages normalization",
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
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--clip_vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
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
        help="The weight decay for the ADAM optimizer used to update the policy network. Common values are 1e-4 - 1e-6.",
    )
    parser.add_argument(
        "--clip_advantages",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--clip_coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
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
    parser.add_argument(
        "--torch_anomaly_detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set torch.autograd.set_detect_anomaly(True) for debbuging",
    )
    parser.add_argument(
        "--skip_step_vlosses",
        action=int,
        default=1,
        help="Number of steps to skip the value loss calculation",
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


# TODO: Check how to solve the seed initialization in the ValueNetwork
# to don't mess up with the sample generation in the DDIMSchedulerA
# of previous experiments.
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
            layer_init(nn.Linear(64, 1), std=1.0),
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

    if args.torch_anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

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

    # NOTE: input shape of 1st layer fixed in the ValueNetwork arquitecture
    value_network = ValueNetwork().to(args.device)

    combined_parameters = list(image_pipe.unet.parameters()) + list(
        value_network.parameters()
    )

    optimizer = torch.optim.Adam(
        combined_parameters,
        weight_decay=args.weight_decay,
    )

    # Resume from ckpt--------------------------------------------------------------
    if args.resume_from_ckpt is not None:
        logging.info("Resume training from ckpt path: %s", args.resume_from_ckpt)
        # Add a descripting message to the wandb
        if args.wandb_logging:
            wandb.run.notes = f"Resuming training from ckpt: {args.resume_from_ckpt}"
        ckpt = torch.load(args.resume_from_ckpt, map_location=torch.device(args.device))
        image_pipe.unet.load_state_dict(ckpt["policy_model_state_dict"])
        value_network.load_state_dict(ckpt["value_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

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
        image_pipe.unet.load_state_dict(ckpt["policy_model_state_dict"])
        value_network.load_state_dict(ckpt["value_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

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
        "Total training steps: %s | warmup steps: %s | lr increment: %s | policy min lr: %s",
        total_training_steps,
        warmup_steps,
        lr_increment,
        min_lr,
    )
    # Training Loop-----------------------------------------------------------------
    track_lrs = []
    mean_rewards = []
    mean_values = []
    mean_returns = []
    mean_advantages = []
    epoch_global_loss = []
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
    start_time = time.time()
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

        (
            all_step_preds,
            log_probs,
            advantages,
            returns,
            all_rewards,
            all_value_estimates,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # (1) Collect data in batches
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

            # compute reward signal for each denoised trajectory, and denoised
            # prediction within the trajectory (DDIM denosied obs)
            batch_trajectory_rewards = []

            for t in range(batch_denoised_all_step_preds.shape[0]):
                batch_trajectory_rewards.append(
                    reward_model(batch_denoised_all_step_preds[t])
                )

            # Concat the rewards into a tensor of shape (T, B)
            batch_trajectory_rewards = torch.stack(batch_trajectory_rewards)

            # Compute return using discounted reward-on-to-go (T, B)
            batch_trajectory_returns = compute_discounted_returns(
                batch_trajectory_rewards, args.gamma
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
                batch_trajectory_returns - batch_value_estimates
            )

            # store information...
            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            returns.append(batch_trajectory_returns)
            advantages.append(batch_trajectory_advantages)
            all_rewards.append(batch_final_rewards)
            all_value_estimates.append(batch_value_estimates)

        # concatenate the collected data
        # ----------------------------------------------------------------------
        all_step_preds = torch.cat(
            all_step_preds, dim=1
        )  # concatenate across batch dim
        log_probs = torch.cat(log_probs, dim=1)  # concatenate across batch dim
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        all_rewards = torch.cat(all_rewards)
        values = torch.cat(all_value_estimates).detach()

        # save the mean and std of reward, value, reeturns, and advantages of the current samples
        mean_rewards.append(all_rewards.mean().item())
        logging.info(" -> mean reward: %s", mean_rewards[-1])
        mean_values.append(values.mean().item())
        logging.info(" -> mean value: %s", mean_values[-1])
        mean_returns.append(returns.mean().item())
        logging.info(" -> mean returns: %s", mean_returns[-1])
        mean_advantages.append(advantages.mean().item())
        logging.info(" -> mean advantages: %s", mean_advantages[-1])

        # Track variance explaind by the value prediction
        # See: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L305C1-L307C86
        y_pred, y_true = values.detach().cpu().numpy(), returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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
                    "mean_returns": mean_returns[-1],
                    "std_returns": returns.std().item(),
                    "mean_advantages": mean_advantages[-1],
                    "std_advantages": advantages.std().item(),
                    "reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy()),
                    "explained_variance": explained_var,
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
        del batch_trajectory_returns
        del batch_trajectory_rewards
        del batch_trajectory_advantages
        del batch_final_rewards
        del batch_value_estimates
        del all_rewards
        # del values
        del all_value_estimates
        flush()

        # ~~ evaluation step ~~
        # Ensure the devset is initialized with the initial model parameters before any updates.
        if args.eval_every_each_epoch is not None and epoch == 0 and args.wandb_logging:
            logging.info(
                "Initializing devset with images and reward trajectories using seed %s, prior to the first update of model parameters at epoch %s.",
                args.eval_rnd_seed,
                epoch + 1,
            )
            eval_imgs, eval_rdf, eval_denoised_rdf_, eval_value_df, _, _ = (
                evaluation_loop(
                    reward_model,
                    value_network,
                    scheduler,
                    image_pipe,
                    args.device,
                    num_samples=args.num_eval_samples,
                    random_seed=args.eval_rnd_seed,
                )
            )
            initial_eval_samples = eval_imgs.detach().cpu().clone()
            initial_eval_trajectories = eval_rdf.copy()
            initial_eval_value_df = eval_value_df.copy()
            initial_eval_denoised_trajectories = eval_denoised_rdf_.copy()
        # ~~ end evaluation step ~~

        # (2) Split data into minibatches and update the parameters (exploitation)
        # ----------------------------------------------------------------------
        # For num_inner_epochs times, we go over each sample compute the loss,
        # backpropagate, and update our diffusion model.
        global_inner_loop_losses = []
        pg_inner_loop_losses = []
        value_inner_loop_losses = []

        logging.info("Starting inner loop...")
        for inner_epoch in progress_bar(range(args.num_inner_epochs)):
            logging.info("Split trajectory data in mini batches")
            # chunk the samples collected into batches
            # all_step_preds and log_probs (T+1, B, C, H, W)
            # and advantages (T, B) and values (T, B)
            all_step_preds_chunked = torch.chunk(
                all_step_preds, args.num_batches, dim=1
            )
            log_probs_chunked = torch.chunk(log_probs, args.num_batches, dim=1)
            # NOTE: why we chunk the advantages and values across dim=0?
            advantages_chunked = torch.chunk(advantages, args.num_batches, dim=0)
            returns_chunked = torch.chunk(returns, args.num_batches, dim=0)
            values_chunked = torch.chunk(values, args.num_batches, dim=0)

            logging.info(
                f"Checking minibatches (mb) shapes:\n->{log_probs_chunked[0].shape}\n->{advantages_chunked[0].shape}\n->{returns_chunked[0].shape}\n->{values_chunked[0].shape}"
            )

            global_loss_value = 0.0
            pg_loss_value = 0.0
            value_loss_value = 0.0

            logging.info("Iterate over the mini batches...")
            # now we start to iterate over the batches (manual dataloader)
            for i in progress_bar(range(len(all_step_preds_chunked))):
                optimizer.zero_grad()
                global_step += 1  # lr counter

                logging.info("Setting policy learning rate given global step")
                if args.initial_lr == args.peak_lr:
                    # Skip the warmup phase and cosine annealing if the initial_lr
                    # is equal to the peak_lr (fix learning rate)
                    lr = args.initial_lr
                    logging.info(
                        "training step %s / %s, policy lr constant: %s",
                        global_step + 1,
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
                            global_step + 1,
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
                            global_step + 1,
                            total_training_steps,
                            lr,
                        )

                # Apply the calculated learning rate to the optimizer
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(optimizer.param_groups[0]["lr"])

                # Iterate through the trajectory by minibatches
                # --------------------------------------------------------------
                mb_x = all_step_preds_chunked[i]
                mb_advantages = advantages_chunked[i]
                mb_returns = returns_chunked[i]
                mb_values = values_chunked[i]
                mb_logprobs = log_probs_chunked[i]
                eta = 1.0  # constant
                loss_value = 0.0
                pg_loss_value = 0.0
                v_loss_value = 0.0
                logr = 0.0
                clipfracs = []

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + EPS
                    )

                for j, t in enumerate(scheduler.timesteps):
                    input = scheduler.scale_model_input(mb_x[j].detach(), t)
                    pred = image_pipe.unet(input, t).sample
                    scheduler_output = scheduler.step(
                        pred,
                        t,
                        mb_x[j].detach(),
                        eta,
                        variance_noise=0,
                    )
                    prev_sample_mean = scheduler_output.prev_sample
                    t_1 = (
                        t
                        - scheduler.config.num_train_timesteps
                        // scheduler.num_inference_steps
                    )
                    variance = scheduler._get_variance(t, t_1)
                    std_dev_t = eta * variance ** (0.5)

                    prev_sample = (
                        prev_sample_mean
                        + torch.randn_like(prev_sample_mean) * std_dev_t
                    )

                    current_log_probs = calculate_log_probs(
                        mb_x[j + 1].detach(),
                        prev_sample_mean,
                        std_dev_t,
                    ).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

                    logratio = current_log_probs - mb_logprobs[j].detach()
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Follow approximation KL(3) based on: http://joschu.net/blog/kl-approx.html
                        # Check also: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L263
                        old_approx_kl = (-logratio).mean().item()
                        approx_kl = ((ratio - 1) - logratio).mean().item()  # k3
                        clipfracs += [
                            ((ratio - 1).abs() > args.clip_coef).float().mean().item()
                        ]

                    # Policy loss (check dim de A y subset con [j])
                    pg_loss1 = -mb_advantages[j].detach() * ratio
                    pg_loss2 = -mb_advantages[j].detach() * torch.clamp(
                        ratio,
                        1 - args.clip_coef,
                        1 + args.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    pg_loss_value += pg_loss.item()

                    # Value loss
                    # estimate the value on the new sample, generate by the
                    # new policy (prev_sample)
                    if j > args.skip_step_vlosses:
                        logging.info("Computing loss with value loss")
                        newvalue = value_network(prev_sample)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - mb_returns[j].detach()) ** 2
                            v_clipped = mb_values[j] + torch.clamp(
                                newvalue - mb_values[j],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - mb_returns[j].detach()) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = (
                                0.5 * ((newvalue - mb_returns[j].detach()) ** 2).mean()
                            )

                        v_loss_value += v_loss.item()

                        # entropy_loss = current_log_probs.mean()
                        # loss = pg_loss - args.ent_coef * entropy_loss + + v_loss * args.vf_coef
                        loss = pg_loss + v_loss * args.vf_coef
                    else:
                        logging.info("Computing loss only with policy loss")
                        loss = pg_loss

                    loss_value += loss.item()
                    loss.backward()

                    # aggregate the log ratio between the current and
                    # original policy
                    logr += torch.sum(
                        current_log_probs - mb_logprobs[j].detach(),
                    )

                torch.nn.utils.clip_grad_norm_(
                    image_pipe.unet.parameters(),
                    max_norm=args.max_grad_norm,
                )  # gradient clipping policy

                torch.nn.utils.clip_grad_norm_(
                    value_network.parameters(),
                    max_norm=args.max_grad_norm,
                )  # gradient clipping value

                optimizer.step()

                if args.wandb_logging:
                    wandb.log(
                        {
                            "loss": loss_value,
                            "policy_loss": pg_loss_value,
                            "value_loss": v_loss_value,
                            # "entropy_loss": entropy_loss,
                            "pct_clipped_ratios": clipfracs,
                            "prob_ratio": wandb.Histogram(
                                ratio.detach().cpu().numpy(),
                            ),
                            "approx_kl": approx_kl,
                            "old_approx_kl": old_approx_kl,
                            "epoch": epoch,
                            "batch": i,
                            "learning_rate": lr,
                            "Steps Per Seconds": int(
                                global_step + 1 / (time.time() - start_time)
                            ),
                        },
                    )
                del mb_x
                del mb_advantages
                del mb_returns
                del mb_values
                del mb_logprobs
                del loss
                del loss_value
                del pg_loss1
                del pg_loss2
                del pg_loss
                del newvalue
                del prev_sample
                del logratio
                del ratio
                del input
                del scheduler_output
                del variance
                del std_dev_t
                del pred
                del prev_sample_mean
                del pg_loss_value
                del v_loss_value
                del logr
                del clipfracs
                flush()

            global_inner_loop_losses.append(global_loss_value / args.num_batches)
            pg_inner_loop_losses.append(pg_loss_value / args.num_batches)
            value_inner_loop_losses.append(value_loss_value / args.num_batches)

            if args.wandb_logging:
                wandb.log(
                    {
                        "global loss": global_inner_loop_losses[-1],
                        "average policy loss": pg_inner_loop_losses[-1],
                        "average value loss": value_inner_loop_losses[-1],
                        "inner_epoch": inner_epoch + 1,
                    },
                )

            logging.info(
                " -> average policy loss: %s | average value loss: %s | average global loss: %s | inner epoch: %s",
                pg_inner_loop_losses[-1],
                value_inner_loop_losses[-1],
                global_inner_loop_losses[-1],
                inner_epoch + 1,
            )

        epoch_global_loss.append(global_inner_loop_losses)
        epoch_policy_loss.append(pg_inner_loop_losses)
        epoch_value_loss.append(value_inner_loop_losses)

        # clean variables
        del all_step_preds
        del log_probs
        del advantages
        del returns
        del pg_inner_loop_losses
        del value_inner_loop_losses
        del global_inner_loop_losses
        flush()

        # Start evaluation loop (each args.eval_every_each_epoch)
        # ----------------------------------------------------------------------
        # TODO: Compute value estimation and advantage in the evaluation loop
        if (
            args.eval_every_each_epoch is not None
            and (((epoch + 1) % args.eval_every_each_epoch) == 0 or epoch == 0)
            and args.wandb_logging
        ):
            logging.info("Evaluating model on epoch %s", epoch + 1)
            eval_imgs, eval_rdf, eval_denoised_rdf, eval_value_df, eval_logp, k = (
                evaluation_loop(
                    reward_model,
                    value_network,
                    scheduler,
                    image_pipe,
                    args.device,
                    num_samples=args.num_eval_samples,
                    random_seed=args.eval_rnd_seed,
                )
            )

            # compute values and advantages
            eval_denoised_rds = eval_denoised_rdf.copy()

            logging.info(
                f"Checking denoised returns shapes: {eval_denoised_rdf.shape}\n -> {eval_denoised_rdf.head(n=3)}"
            )

            logging.info(
                f"Checking eval shapes: {eval_value_df.shape}\n -> {eval_value_df.head(n=3)}"
            )

            # log the evaluation results in a wandb.Table
            table = wandb.Table(
                columns=[
                    "original_samples",
                    "current_samples",
                    "current_final_reward",
                    "original_final_reward",
                    "diff_reward",
                    "current_value_estimate",
                    "original_value_estimate",
                    "reward_trajectory",
                ],
            )

            for (
                o_img,
                c_img,
                rc,
                drc,
                vrc,
            ) in zip(
                initial_eval_samples,
                eval_imgs,
                eval_rdf,
                eval_denoised_rdf,
                eval_value_df,
            ):
                # create reward plot trajectory
                plt.figure(figsize=(10, 4))
                plt.plot(
                    eval_denoised_rdf[rc],
                    color="indigo",
                    label="current denoised rwd trajectory",
                )
                plt.plot(
                    eval_rdf[rc],
                    color="mediumseagreen",
                    label="current rwd trajectory",
                )
                plt.plot(
                    initial_eval_denoised_trajectories[rc],
                    color="lightgrey",
                    label="initial denoised rwd trajectory",
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
                    eval_rdf[rc][-1:].item(),  # current final reward
                    initial_eval_trajectories[rc][-1:].item(),  # initial final reward
                    eval_rdf[rc][-1:].item()  # diff reward
                    - initial_eval_trajectories[rc][-1:].item(),
                    eval_value_df[vrc][-1:].item(),  # current value estimate
                    initial_eval_value_df[vrc][-1:].item(),  # initial value estimate
                    wandb.Image(
                        plt,
                    ),
                )
            wandb.log({"eval_table": table}, commit=False)
            plt.close()
            eval_mean_reward = eval_rdf.iloc[-1, :].mean()
            eval_value_mean = eval_value_df.iloc[-1, :].mean()
            logging.info(
                " -> eval mean reward | eval value reward (%s epoch): %s | %s",
                epoch + 1,
                eval_mean_reward,
                eval_value_mean,
            )
            wandb.log(
                {
                    "eval_mean_reward": eval_mean_reward,
                    "eval_value_df_mean": eval_value_mean,
                }
            )
            del eval_imgs
            del eval_rdf
            del eval_denoised_rdf
            del eval_value_df
            del eval_logp
            del eval_value_mean
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
                        "policy_model_state_dict": image_pipe.unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "value_model_state_dict": value_network.state_dict(),
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
        del returns_chunked

        flush()

    if args.wandb_logging:
        wandb.finish()
