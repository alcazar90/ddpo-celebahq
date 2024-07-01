"""Implementation of the DDPO algorithm."""

import logging

import pandas as pd
import torch

from ddpo.config import EPS
from ddpo.sampling import calculate_log_probs, sample_from_ddpm_celebahq

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_discounted_returns(rewards, gamma=0.99):
    # Compute return using discounted reward-on-to-go (T, B)
    T, B = rewards.shape
    returns = torch.zeros_like(rewards)

    # Initialize reward-to-go for the last time step
    returns[-1] = rewards[-1]

    # Iterate over each timestep in reverse order to accumulate discounted rewards
    for i, t in enumerate(reversed(range(T - 1))):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def compute_loss(
    x_t,
    original_log_probs,
    advantages,
    returns,
    values,
    clip_advantages,
    clip_coef,
    image_pipe,
    scheduler,
    value_function,
    vf_coef,
    ent_coef,
    clip_vloss=True,
    norm_adv=True,
    device="cuda",
    eta=1,
):
    """Compute DDPO_is loss for a batch of samples.

    Args:
        x_t (torch.Tensor): The input samples for the current timestep.
        original_log_probs (torch.Tensor): The log probabilities of the original policy.
        advantages (torch.Tensor): The advantages for each sample.
        returns (torch.Tensor): The returns for each sample.S
        values (torch.Tensor): The values for each sample.
        clip_advantages (float): The maximum value to clip the advantages.
        clip_coef (float): The maximum value to clip the ratio.
        image_pipe (ImagePipe): The image processing pipeline.
        scheduler (Scheduler): The scheduler for the DDPO algorithm.
        value_function (torch.nn.Module): The value function to use for computing the advantages.
        vf_coef (float): The coefficient for the value function loss.
        ent_coef (float): The coefficient for the entropy loss.
        norm_adv (bool, optional): Whether to normalize the advantages. Defaults to True.
        device (torch.device): The device to perform computations on.
        eta (float, optional): The scaling factor for the standard deviation. Defaults to 1.

    Returns:
        float: The computed loss value.
        float: The computed policy loss value.
        float: The computed value loss value.
        torch.Tensor: The ratio of probabilities between the current policy and the original policy.
        float: The percentage of clipped ratios.
        float: The mean KL divergence between the current policy and the original policy.

    """
    unet = image_pipe.unet.to(device)
    num_inference_steps = scheduler.num_inference_steps
    pg_loss_value = 0.0
    value_loss_value = 0.0
    entropy_loss_value = 0.0
    logr = 0.0

    for i, t in enumerate(scheduler.timesteps):
        if norm_adv:
            adv = (advantages[i] - advantages[i].mean()) / (advantages[i].std() + EPS)
        else:
            adv = advantages[i]

        clipped_advantages = torch.clip(
            adv,
            -clip_advantages,
            clip_advantages,
        ).detach()

        # scale the input by the current timestep t and predict the noise residual
        input = scheduler.scale_model_input(x_t[i].detach(), t)
        pred = unet(input, t).sample

        # compute the "previous" noisy sample mean and variance, and get log probs
        scheduler_output = scheduler.step(
            pred,
            t,
            x_t[i].detach(),
            eta,
            variance_noise=0,
        )

        # this is the mean and not full sample since variance is 0
        prev_sample_mean = scheduler_output.prev_sample

        # compute variance between two timesteps, considering the jumps
        # between training and inference timesteps
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)

        # generate new samples using reparametrization trick (adding noise)
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        # compute log probs of the new sample and the original policy
        current_log_probs = calculate_log_probs(
            x_t[i + 1].detach(),
            prev_sample_mean,
            std_dev_t,
        ).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

        # importance ratio of the new policy to the old policy
        ratio = torch.exp(current_log_probs - original_log_probs[i].detach())

        # compute entropy loss
        # entropy = -torch.sum(torch.exp(current_log_probs) * current_log_probs, dim=-1)
        # entropy_loss = entropy.mean()
        # entropy_loss_value += entropy_loss.item()

        # policy loss
        pg_loss1 = -clipped_advantages * ratio  # this is the surrogate loss
        pg_loss2 = -clipped_advantages * torch.clamp(
            ratio,
            1.0 - clip_coef,
            1.0 + clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        pg_loss_value += pg_loss.item()

        # compute % of clipped ratios
        pct_clipped_ratios = (
            torch.sum(
                torch.logical_or(ratio < 1.0 - clip_coef, ratio > 1.0 + clip_coef),
            )
            / ratio.size(0)
        ).item()

        # calculate KL between the current policy and the original policy
        logr += torch.sum(
            current_log_probs - original_log_probs[i].detach(),
        )

        # compute new values based on denoised predictin (DDIM Eq.9)
        # VERIFY: this is the correct way to compute the denoised trajectory?
        # using the states from the original policy?
        alpha_prod_t = image_pipe.scheduler.alphas_cumprod[t].detach()
        denoised_prev_sample = (
            x_t[i] - torch.sqrt(1 - alpha_prod_t) * prev_sample.detach()
        ) / torch.sqrt(alpha_prod_t)

        # value loss
        # See: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L280
        value = value_function(denoised_prev_sample)
        if clip_vloss:
            v_loss_unclipped = (value - returns[i].detach()) ** 2
            v_clipped = values[i] + torch.clamp(
                value - values[i],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - returns[i].detach()) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            # Compute value loss for the current timestep
            v_loss = 0.5 * ((value - returns[i].detach()) ** 2).mean()

        value_loss_value += v_loss.item()

        # loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
        loss = pg_loss + v_loss * vf_coef
        loss.backward()

    # Follow approximation KL(3) based on: http://joschu.net/blog/kl-approx.html
    # Check also: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L263
    old_approx_kl = (-logr).mean().item()
    approx_kl = ((logr.exp() - 1) - logr).mean().item()  # k3

    return (
        loss.item(),
        pg_loss_value,
        value_loss_value,
        entropy_loss_value,
        ratio,
        pct_clipped_ratios,
        approx_kl,
        old_approx_kl,
    )


@torch.no_grad()
def evaluation_loop(
    reward_function,
    value_function,
    scheduler,
    image_pipe,
    device="cuda",
    num_samples: int = 4,
    random_seed: int = 666,
    previous_logp=None,
):
    """Given a random seed compute and return metrics to evaluate on same same subset the model

    Args:
        reward_function: The reward function used to compute rewards over the trajectory.
        value_function: The value function represent the model to estimate the value.
        scheduler: The scheduler used for sampling from the model.
        image_pipe: The image pipeline used for processing images.
        device: The device (CPU or GPU) to perform computations on.
        num_samples (optional): The number of samples to generate from the model. Defaults to 4.
        random_seed (optional): The random seed to use for reproducibility. Defaults to 666.
        previous_logp (optional): The log probabilities of the previous sample set. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - The last trajectory sample.
            - A DataFrame containing the rewards for each sample at each timestep.
            - The log probabilities of the current sample set.
            - The KL divergence between the current and previous sample sets.
    """
    # (1) Obtain sample and latents
    with torch.random.fork_rng():
        trajectory, denoised_trajectory, logp = sample_from_ddpm_celebahq(
            num_samples=num_samples,
            scheduler=scheduler,
            image_pipe=image_pipe,
            device=device,
            random_seed=random_seed,
        )

    # (2) Compute reward over raw and denoised trajectory
    r = [reward_function(t) for t in trajectory]
    rd = [reward_function(t) for t in denoised_trajectory]
    # rows are t in trajectory, columns are samples ids, and values are the
    # corresponding rewards for the given sample at timestep t
    r_df = pd.DataFrame(torch.vstack(r).detach().cpu().numpy())
    rd_df = pd.DataFrame(torch.vstack(rd).detach().cpu().numpy())

    # (3) Compute the KL with the previous sample set
    k = None
    if previous_logp is not None:
        logr = (previous_logp - logp).sum(axis=0)
        k = (logr.exp() - 1) - logr
        k = k.mean().item()

    # (4) TODO: Compute value function over denoised trajectory
    value_function.eval()
    with torch.no_grad():
        v = [value_function(t) for t in denoised_trajectory]
    v_df = pd.DataFrame(torch.vstack(v).detach().cpu().numpy())
    value_function.train()

    # (5) TODO: Compute discounted returns. Useful to have a function for this

    # (6) TODO: Compute advantages. Required the returns...

    # (7) Return everything...
    return trajectory[-1], r_df, rd_df, v_df, logp.detach().cpu(), k
