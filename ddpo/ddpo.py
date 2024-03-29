"""Implementation of the DDPO algorithm."""

import math

import torch
from tqdm import tqdm

from ddpo.utils import flush


def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


EPS = 1e-6


def standardize(x):
    """Standardizes the given input array.

    Args:
    ----
    x (array-like): The input array to be standardized.

    Returns:
    -------
    array-like: The standardized array.

    """
    return (x - x.mean()) / (x.std() + EPS)


def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t, eps=EPS):
    """Compute logs probs for prev_sample from a normal distribution with mean
    prev_sample_meand and std std_dev_t.
    """
    std_dev_t = torch.clip(std_dev_t, eps)
    return (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t**2)
        - torch.log(std_dev_t)
        - math.log(math.sqrt(2 * math.pi))
    )


@torch.no_grad()
def sample_from_ddpm_celebahq(
    num_samples,  # noqa: ANN001
    scheduler,
    image_pipe,
    device,
    eta=1,
    random_seed=None,
):
    """Sample a batch of images from the google/ddpm-celebahq-256 model using a specified scheduler, image pipeline, reward model, and device.

    Reference of diffuser sample loop: https://huggingface.co/blog/stable_diffusion

    Args:
    ----
    num_samples (int): The number of samples to generate.
    scheduler (DDIMScheduler): The scheduler object that controls the sampling process.
    image_pipe (ImagePipeline): The image pipeline object used for processing images.
    device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform computations.
    random_seed (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
    -------
    tensor: A tensor containing the trajectories of the entire batach (T, B, C, H, w).
    tensor: A tensor containing the log probabilities of the trajectories (T, B).

    """
    if random_seed:
        torch.manual_seed(random_seed)

    num_inference_steps = scheduler.num_inference_steps

    # initialize a batch of random noise
    xt = torch.randn(num_samples, 3, 256, 256).to(device)  # (B, C, H, W)

    # save initial state x_T and intermediate steps, saave log_probs for the trajectory
    trajectory, log_probs = [xt], []

    for _, t in enumerate(progress_bar(scheduler.timesteps)):
        # [S] scale input based on the timestep
        model_input = scheduler.scale_model_input(xt, timestep=t)

        # [S] get the noise prediction (unet predicts noise residual)
        noise_pred = image_pipe.unet(model_input, t).sample

        # [S] using the prediction noise we can predict the denoised image representation
        # compute the "previous" noisy sample mean
        scheduler_output = scheduler.step(noise_pred, t, xt, eta, variance_noise=0)
        prev_sample_mean = (
            scheduler_output.prev_sample
        )  # this is the mean and not full sample since variance is 0

        # [S] Computa la varianza entre los do timesteps actual y anterior,
        # se debe considerar los saltos entree timesteps de entrenamiento e inferencia.
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)

        # [S] generamos nuevas muestras (re-parametrization trick)
        prev_sample = (
            prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
        )  # get full sample by adding noise

        # [S] compute the log probs of the new sample
        log_probs.append(
            calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(
                dim=tuple(range(1, prev_sample_mean.ndim)),
            ),
        )

        trajectory.append(prev_sample)
        xt = prev_sample

    # now we will release the VRAM memory deleting the variable bounded to the VRAM
    # and use flush()
    del xt
    del model_input
    del noise_pred
    del scheduler_output
    del prev_sample_mean
    del prev_sample
    del variance
    del std_dev_t
    del num_inference_steps
    flush()

    # The dimensions of the tensor are: (T+1, B, C, H, W), (T, B), (B, 1)
    return torch.stack(trajectory), torch.stack(log_probs)


def compute_loss(
    x_t,
    original_log_probs,
    advantages,
    clip_advantages,
    clip_ratio,
    image_pipe,
    scheduler,
    device,
    eta=1,
):
    """Compute DDPO_is loss for a batch of samples.

    Args:
    ----
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
    -------
      float: The computed loss value.

    """
    unet = image_pipe.unet.to(device)
    num_inference_steps = scheduler.num_inference_steps
    loss_value = 0.0
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
        ratio = torch.exp(
            current_log_probs - original_log_probs[i].detach(),
        )  # this is the importance ratio of the new policy to the old policy
        unclipped_loss = -clipped_advantages * ratio  # this is the surrogate loss
        clipped_loss = -clipped_advantages * torch.clip(
            ratio,
            1.0 - clip_ratio,
            1.0 + clip_ratio,
        )  # this is the surrogate loss, but with artificially clipped ratios
        loss = torch.max(
            unclipped_loss,
            clipped_loss,
        ).mean()  # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
        loss.backward()  # perform backward here, gets accumulated for all the timesteps

        loss_value += loss.item()
    return loss_value
