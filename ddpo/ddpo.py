"""Implementation of the DDPO algorithm."""

import logging

import pandas as pd
import torch

from ddpo.sampling import calculate_log_probs, sample_from_ddpm_celebahq

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_loss(
    x_t,
    original_log_probs,
    advantages,
    returns,
    clip_advantages,
    clip_ratio,
    image_pipe,
    scheduler,
    value_function,
    device="cuda",
    eta=1,
):
    """Compute DDPO_is loss for a batch of samples.

    Args:
        x_t (torch.Tensor): The input samples for the current timestep.
        original_log_probs (torch.Tensor): The log probabilities of the original policy.
        advantages (torch.Tensor): The advantages for each sample.
        returns (torch.Tensor): The returns for each sample.S
        clip_advantages (float): The maximum value to clip the advantages.
        clip_ratio (float): The maximum value to clip the ratio.
        image_pipe (ImagePipe): The image processing pipeline.
        scheduler (Scheduler): The scheduler for the DDPO algorithm.
        value_function (torch.nn.Module): The value function to use for computing the advantages.
        device (torch.device): The device to perform computations on.
        eta (float, optional): The scaling factor for the standard deviation. Defaults to 1.

    Returns:
        float: The computed policy loss value.
        float: The computed value loss value.
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

    new_values = []
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

        # compute new values based on denoised predictin (DDIM Eq.9)
        # VERIFY: this is the correct way to compute the denoised trajectory?
        # using the states from the original policy?
        alpha_prod_t = image_pipe.scheduler.alphas_cumprod[t]
        denoised_prev_sample = (
            x_t[i] - torch.sqrt(1 - alpha_prod_t) * prev_sample
        ) / torch.sqrt(alpha_prod_t)

        # update the trajectory
        new_values.append(denoised_prev_sample)

    logging.info(
        f"new_values length: {len(new_values)}, shape of element: {new_values[0].shape}",
    )
    logging.info(
        f"returns shape: {returns.shape}",
    )

    # concatenate new denoised states and compute new values
    new_values = torch.stack(new_values)

    logging.info(
        f"new_values shape: {new_values.shape}",
    )

    mb_new_values = value_function(torch.stack(new_values)).view(-1)
    mb_returns = returns.view(-1)

    # compute the value loss
    # TODO: add option to clipped version of value loss
    # See: https://github.com/vwxyzjn/ppo-implementation-details/blob/fbef824effc284137943ff9c058125435ec68cd3/ppo.py#L280
    value_loss = 0.5 * ((mb_new_values - mb_returns) ** 2).mean()

    value_loss.backward()

    # Follow approximation KL based on: http://joschu.net/blog/kl-approx.html
    k3 = (logr.exp() - 1) - logr
    return pg_loss_value, value_loss.item(), ratio, pct_clipped_ratios, k3.mean().item()


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
        trajectory, _, logp = sample_from_ddpm_celebahq(
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
