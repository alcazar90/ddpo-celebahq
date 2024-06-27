"""Implementation of the DDPO algorithm."""

import pandas as pd
import torch

from ddpo.sampling import calculate_log_probs, sample_from_ddpm_celebahq


def compute_loss(
    x_t,
    original_log_probs,
    advantages,
    clip_advantages,
    clip_ratio,
    image_pipe,
    scheduler,
    device="cuda",
    eta=1,
):
    """Compute DDPO_is loss for a batch of samples.

    Args:
        x_t (torch.Tensor): The input samples for the current timestep.
        original_log_probs (torch.Tensor): The log probabilities of the original policy.
        advantages (torch.Tensor): The advantages for each sample.
        clip_advantages (float): The maximum value to clip the advantages.
        clip_ratio (float): The maximum value to clip the ratio.
        image_pipe (ImagePipe): The image processing pipeline.
        scheduler (Scheduler): The scheduler for the DDPO algorithm.
        device (torch.device): The device to perform computations on.
        eta (float, optional): The scaling factor for the standard deviation. Defaults to 1.

    Returns:
        float: The computed loss value.
        torch.Tensor: The ratio of probabilities between the current policy and the original policy.
        float: The percentage of clipped ratios.
        float: The mean KL divergence between the current policy and the original policy.

    """
    # TODO: captur clipped fraction like in:
    # See: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L252
    unet = image_pipe.unet.to(device)
    num_inference_steps = scheduler.num_inference_steps
    pg_loss_value = 0.0
    logr = 0.0
    for i, t in enumerate(scheduler.timesteps):
        clipped_advantages = torch.clip(
            advantages,
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
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = scheduler_output.prev_sample
        current_log_probs = calculate_log_probs(
            x_t[i + 1].detach(),
            prev_sample_mean,
            std_dev_t,
        ).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

        # calculate loss
        # ratio probability current policy / probability original policy
        ratio = torch.exp(
            current_log_probs - original_log_probs[i].detach(),
        )  # this is the importance ratio of the new policy to the old policy
        unclipped_loss = -clipped_advantages * ratio  # this is the surrogate loss
        clipped_loss = -clipped_advantages * torch.clip(
            ratio,
            1.0 - clip_ratio,
            1.0 + clip_ratio,
        )  # this is the surrogate loss, but with artificially clipped ratios

        # compute % of clipped ratios
        pct_clipped_ratios = (
            torch.sum(
                torch.logical_or(ratio < 1.0 - clip_ratio, ratio > 1.0 + clip_ratio),
            )
            / ratio.size(0)
        ).item()

        pg_loss = torch.max(
            unclipped_loss,
            clipped_loss,
        ).mean()  # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch

        pg_loss.backward()  # perform backward here, gets accumulated for all the timesteps

        pg_loss_value += pg_loss.item()

        # calculate KL between the current policy and the original policy
        logr += torch.sum(
            current_log_probs - original_log_probs[i].detach(),
        )

    # Follow approximation KL based on: http://joschu.net/blog/kl-approx.html
    k3 = (logr.exp() - 1) - logr
    return pg_loss_value, ratio, pct_clipped_ratios, k3.mean().item()


@torch.no_grad()
def evaluation_loop(
    reward_function,
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
        trajectory, logp = sample_from_ddpm_celebahq(
            num_samples=num_samples,
            scheduler=scheduler,
            image_pipe=image_pipe,
            device=device,
            random_seed=random_seed,
        )

    # (2) Compute reward over trajectory
    r = [reward_function(t) for t in trajectory]
    # rows are t in trajectory, columns are samples ids, and values are the
    # corresponding rewards for the given sample at timestep t
    r_df = pd.DataFrame(torch.vstack(r).detach().cpu().numpy())

    # (3) Compute the KL with the previous sample set
    k = None
    if previous_logp is not None:
        logr = (previous_logp - logp).sum(axis=0)
        k = (logr.exp() - 1) - logr
        k = k.mean().item()

    # (4) Return everything...
    return trajectory[-1], r_df, logp.detach().cpu(), k
