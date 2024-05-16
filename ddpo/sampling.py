"""Sampling functions for the google/ddpm-celebahq-256 model."""

import math

import torch
from tqdm.auto import tqdm
import numpy as np

from ddpo.config import EPS
from ddpo.utils import flush


def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def calculate_log_probs(
    prev_sample,
    prev_sample_mean,
    std_dev_t,
    eps=EPS,
):
    """Compute logs probs for prev_sample from a normal distribution with mean
    prev_sample_mean and std std_dev_t.d.

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
def improved_sample_from_ddpm_celebahq(
    num_samples,  # noqa: ANN001
    scheduler,
    image_pipe,
    device,
    epoch,
    num_epochs,
    mean_zone_interest_sampling,
    num_of_segments,
    clusters,
    n_iters,
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
    # initiate_train_steps = sample_timesteps(num_samples, len(scheduler.timesteps), epoch, num_epochs,scheduler, mean_zone_interest_sampling, 40, 39, 41, device)
    initiate_train_steps  = sample_from_clusters(num_samples,epoch, num_epochs,clusters,n_iters, device)
    # (num_samples, current_iteration, total_epochs, clusters, n_iters, device)

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
    return torch.stack(trajectory), torch.stack(log_probs), initiate_train_steps

@torch.no_grad()
def improved_sample_from_ddpm_celebahq_initial_step(
    num_samples,  # noqa: ANN001
    scheduler,
    image_pipe,
    device,
    epoch,
    num_epochs,
    mean_zone_interest_sampling,
    num_of_segments,
    clusters,
    n_iters,
    initial_sample_full_image,
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
        initial_sample_full_image (image): final sample to obtain noisy version and start sampling from it

    Returns:
    -------
        tensor: A tensor containing the trajectories of the entire batach (T, B, C, H, W).
        tensor: A tensor containing the log probabilities of the trajectories (T, B).

    """
    if random_seed:
        torch.manual_seed(random_seed)
    
    initiate_train_steps  = sample_from_clusters(num_samples,epoch, num_epochs,clusters,n_iters, device)
    
    num_inference_steps = scheduler.num_inference_steps

    # initialize a batch of random noise
    # Initialize the tensor to store all noisy images
    xt = torch.zeros((num_samples, 3, 256, 256)).to(device)

    # Generate the noisy images
    for i in range(num_samples):
        noisy_image = add_noise_to_image(initial_sample_full_image, int(scheduler.timesteps[initiate_train_steps]), scheduler)
        xt[i] = noisy_image
    # save initial state x_T and intermediate steps, saave log_probs for the trajectory
    trajectory, log_probs = [], []

    for _ in range(len(scheduler.timesteps)- initiate_train_steps - 1):
        empty_sample = torch.zeros_like(xt).cpu()
        trajectory.append(empty_sample)
    
    trajectory.append(xt)
    
    for _ in range(len(scheduler.timesteps)-initiate_train_steps):
        empty_log_prob = torch.zeros((num_samples,)).cpu()
        log_probs.append(empty_log_prob)
    # initiate_train_steps = sample_timesteps(num_samples, len(scheduler.timesteps), epoch, num_epochs,scheduler, mean_zone_interest_sampling, 40, 39, 41, device)
    # (num_samples, current_iteration, total_epochs, clusters, n_iters, device)

    for i in range(initiate_train_steps, len(scheduler.timesteps)):
        t = scheduler.timesteps[i]
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
    return torch.stack(trajectory), torch.stack(log_probs), initiate_train_steps

@torch.no_grad()
def sample_data_from_celebahq(
    num_samples,
    scheduler,
    image_pipe,
    device,
    random_seed=None,
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
        trajectory.append(xt.clone().detach().cpu())
        # if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        #     trajectory.append(xt.clone().detach().cpu())

    # save trajectories in obs' dictionary
    obs["trajectory"] = trajectory

    # now we will release the VRAM memory deleting the variable bounded to the VRAM and use flush()
    del xt
    del noise_pred
    del model_input
    del scheduler_output
    flush()

    return obs


def sample_timesteps(num_samples, num_timesteps, current_iteration, target_iteration, scheduler, mu=30, variance=40, min_clip=39, max_clip=41, device='cpu'):
    sigma = variance  # Standard deviation for the Gaussian distribution
    gamma = current_iteration / target_iteration  # Blend ratio

    # Full range timesteps for Gaussian probabilities
    full_timesteps = torch.arange(0, num_timesteps, device=device, dtype=torch.float32)
    gaussian_probs = torch.exp(-0.5 * ((full_timesteps - mu) ** 2) / sigma ** 2)
    gaussian_probs /= gaussian_probs.sum()  # Normalize

    # Restricted range timesteps for Uniform probabilities
    uniform_probs = torch.zeros(num_timesteps, device=device)
    uniform_probs[min_clip:max_clip + 1] = 1.0 / (max_clip - min_clip + 1)
    uniform_probs /= uniform_probs.sum()  # Normalize within restricted range

    # Mixed distribution probabilities
    mixed_probs = (1 - gamma) * uniform_probs + gamma * gaussian_probs
    mixed_probs /= mixed_probs.sum()  # Ensure normalization

    # Sampling timesteps based on mixed probabilities
    sampled_timesteps = torch.multinomial(mixed_probs, num_samples, replacement=True)

    # Compute and return the mean of the sampled timesteps
    mean_sampled_timesteps = sampled_timesteps.float().mean().item()
    mean_sampled_timesteps = max(0, min(mean_sampled_timesteps, len(scheduler.timesteps) - 1))

    return int(len(scheduler.timesteps)) - int(mean_sampled_timesteps)


def sample_from_segments(num_samples, num_timesteps, current_iteration, target_iteration, device, num_segments=10):
    # Calculate the size of each segment
    segment_size = num_timesteps // num_segments
    
    # Determine which segment to sample from based on the current iteration
    iterations_per_segment = target_iteration // num_segments
    current_segment_index = current_iteration // iterations_per_segment
    current_segment_index = min(current_segment_index, num_segments - 1)  # Ensure it doesn't go out of range
    
    # Define the start and end of the current segment
    segment_start = current_segment_index * segment_size
    segment_end = segment_start + segment_size
    
    # Create a tensor of timesteps for the current segment
    segment_timesteps = torch.arange(segment_start, segment_end, device=device)
    
    # Sample from the uniform distribution over the current segment
    sampled_indices = torch.randint(len(segment_timesteps), (num_samples,))  # Get random indices
    sampled_timesteps = segment_timesteps[sampled_indices]

    # Convert the sampled timesteps to a list of integers
    # sampled_timesteps_list = [int(x.item()) for x in sampled_timesteps]  # Convert each tensor element to int
    sampled_timesteps_list = [sampled_timesteps[i].item() for i in range(sampled_timesteps.size(0))]
    print("sampled timestep list",sampled_timesteps_list)
    
    return sampled_timesteps_list[0]


def sample_from_clusters(num_samples, current_iteration, total_epochs, clusters, n_iters, device):
    assert sum(n_iters) == total_epochs, "Total sum of n_iters must be equal to total_epochs"
    assert len(n_iters) == len(clusters), "Length of n_iters must be equal to length of clusters"

    # Compute index_iters based on the example given on the whiteboard
    index_iters = []
    for i, times in enumerate(n_iters):
        index_iters.extend([i] * times)

    # Select the cluster based on the current epoch using index_iters
    cluster_index = index_iters[current_iteration]
    cluster = clusters[cluster_index]

    # Determine the format of the cluster and create a tensor of timesteps
    if isinstance(cluster, tuple):
        # Tuple format: represents a range (start, end)
        segment_start, segment_end = cluster
        segment_timesteps = torch.arange(segment_start, segment_end, device=device)
    elif isinstance(cluster, list):
        # List format: explicit list of timesteps
        segment_timesteps = torch.tensor(cluster, device=device)
    else:
        raise ValueError("Cluster format not recognized. Must be a tuple or a list.")

    # Sample from the uniform distribution over the current segment
    sampled_indices = torch.randint(len(segment_timesteps), (num_samples,))
    sampled_timesteps = segment_timesteps[sampled_indices]

    return sampled_timesteps[0].item()  # Return the first value of the sampling list

def add_noise_to_image(x_0, t, scheduler):
    # Get the alphas from the scheduler
    alpha_t = scheduler.alphas_cumprod[t]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Sample noise from a standard normal distribution
    noise = torch.randn_like(x_0)

    # Compute x_t using the equation: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
    x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
    return x_t
# # Test parameters
# clusters = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
# n_iters = [5, 0, 5, 0, 5]
# total_epochs = 15
# num_steps = 10  # New number of samples
# test_results = []

# # Run the test for each epoch
# for current_epoch in range(total_epochs):
#     first_sampled_timestep = initial_step_sampling(num_steps, current_epoch, total_epochs, clusters, n_iters, device='cpu')
#     test_results.append(f"Epoch {current_epoch}: First Sampled Timestep - {first_sampled_timestep}")

# test_results