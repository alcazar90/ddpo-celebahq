"""Sampling functions for the google/ddpm-celebahq-256 model."""

import torch
from tqdm.auto import tqdm

from ddpo.ddpo import calculate_log_probs
from ddpo.utils import flush


def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


@torch.no_grad()
def sample_from_ddpm_celebahq(
    num_samples,  # noqa: ANN001
    scheduler,
    image_pipe,
    device,
    eta=1,
    random_seed=None,
):
    """Sample a batch of trajectories from the google/ddpm-celebahq-256 model using a specified scheduler, image pipeline, reward model, and device.
    Compute their log probabilities at each timestep and return the trajectories and log probabilities.

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
    tensor: A tensor containing the trajectories of the entire batach (T, B, C, H, W).
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

        # [S] compute variance between two timesteps, considering the jumps between training and inference timesteps
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)

        # [S] generate new samples using re-parametrization trick
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

    # now we will release the VRAM memory deleting the variable bounded to the VRAM and use flush()
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


@torch.no_grad()
def sample_data_from_celebahq(
    num_samples, scheduler, image_pipe, device, random_seed=None
):
    """Sample a batch of trajectories from the google/ddpm-celebahq-256 model using a specified scheduler, image pipeline, reward model, and device.
    Save only a summary of each trajectories and their corresponding seeds.

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
    dict: Containing in "seed" the rnd_state used for generate samples' trajectories, and in "trajectory", a tensor containing the trajectories of the entire batch (T, B, C, H, W).
    """
    obs = {}

    if random_seed is not None:
        obs["seed"] = random_seed
        torch.manual_seed(random_seed)

    # initialize a batch of random noise
    xt = torch.randn(num_samples, 3, 256, 256).to(device)

    trajectory = [xt.clone().detach().cpu()]

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # scale input based on the timestep
        model_input = scheduler.scale_model_input(xt, t)

        # get the noise prediction
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # compute the update sample regarding the scheduler
        scheduler_output = scheduler.step(noise_pred, t, xt)

        # update x
        xt = (
            scheduler_output.prev_sample
        )  # .prev_sample attribute refer to the backward process (denoising)

        # save tensors each 10 steps and in the last
        if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
            trajectory.append(xt.clone().detach().cpu())

    # save trajectories in obs' dictionary
    obs["trajectory"] = trajectory

    # now we will release the VRAM memory deleting the variable bounded to the VRAM and use flush()
    del xt
    del noise_pred
    del model_input
    del scheduler_output
    flush()

    return obs
