import torch
from tqdm import tqdm
from ddpo.utils import flush
from ddpo.sampling import add_noise_to_image

def predict_denoised_image(image_pipe, x_t, t, scheduler, device='cuda'):
    """
    Estimate the denoised image from a noisy image at a specific timestep using the model within image_pipe.

    Parameters:
        image_pipe (ImagePipeline): Contains the UNet model used for predicting the noise.
        x_t (torch.Tensor): The noisy image tensor at timestep t.
        t (int): The current timestep index.
        scheduler (Scheduler): The scheduler used for controlling the sampling process.
        device (str): The device to perform computation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The estimated denoised image tensor.
    """
    # Ensure x_t is on the correct device
    x_t = x_t.to(device)

    # Ensure the tensor has four dimensions [B, C, H, W]
    if x_t.dim() != 4:
        raise ValueError("Input tensor must have four dimensions [B, C, H, W]")

    # Get the necessary parameters for the timestep
    alpha_t = scheduler.alphas_cumprod[t].to(device)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Model evaluation mode and predict noise
    image_pipe.unet.eval()
    with torch.no_grad():
        model_input = scheduler.scale_model_input(x_t, t)
        noise_estimate = image_pipe.unet(model_input, t)["sample"]

    # Estimate the denoised image
    x_0_hat = (x_t - sqrt_one_minus_alpha_t * noise_estimate) / torch.sqrt(alpha_t)

    return x_0_hat


def predict_and_store_denoised_images_in_batches(images, image_pipe, scheduler, device='cuda'):
    """
    Process a list of batches of images, predict the denoised images for each, and store them.

    Parameters:
        images (list): A list of batches, where each batch contains one or more images of shape [3, 256, 256].
        image_pipe (ImagePipeline): Contains the UNet model used for predicting the noise.
        scheduler (Scheduler): The scheduler used for controlling the sampling process.
        device (str): The device to perform computation on ('cuda' or 'cpu').

    Returns:
        list: A list of lists containing the denoised images for each batch.
    """
    denoised_batches = []

    # Iterate over each batch
    for batch in tqdm(images, desc="Processing batches"):
        denoised_images = []
        # Each batch has 2 images
        for img in batch:
            # Ensure the image tensor is properly formatted as [B, C, H, W]
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # Add batch dimension if needed

            # Denoise each image at each timestep
            denoised_per_timestep = []
            for t in scheduler.timesteps:
                denoised_img = predict_denoised_image(image_pipe, img_tensor, t, scheduler, device)
                denoised_per_timestep.append(denoised_img.cpu().detach())  # Detach and move to CPU

            denoised_images.append(denoised_per_timestep)
        denoised_batches.append(denoised_images)

    return denoised_batches


@torch.no_grad()
def sample_denoised_data_from_celebahq(
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
    trajectory_denoised = [xt.clone().detach().cpu()]

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
        x_0_hat = (
            scheduler_output.pred_original_sample
        )
        trajectory.append(xt.clone().detach().cpu())
        trajectory_denoised.append(x_0_hat.clone().detach().cpu())
        # if t != scheduler.timesteps[-1]:
        #     trajectory.append(xt.clone().detach().cpu())
        #     # Compute the predicted denoised image at this step
        #     x_0_hat = predict_denoised_image(image_pipe, xt, t, scheduler, device)
        #     trajectory_denoised.append(x_0_hat.clone().detach().cpu())
        # save tensors each 10 steps and in the last

        # if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        #     trajectory.append(xt.clone().detach().cpu())

    # save trajectories in obs' dictionary
    obs["trajectory"] = trajectory
    obs["trajectory_denoised"] = trajectory_denoised

    # now we will release the VRAM memory deleting the variable bounded to the VRAM and use flush()
    del xt
    del noise_pred
    del model_input
    del scheduler_output
    flush()

    return obs

@torch.no_grad()
def sample_denoised_data_from_celebahq_intermediate_step(
    num_samples,
    scheduler,
    image_pipe,
    device,
    initial_step,
    final_image_original,
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

    # Generate the noisy images
    for i in range(num_samples):
        noisy_image = add_noise_to_image(final_image_original, int(scheduler.timesteps[initial_step]), scheduler)
        xt[i] = noisy_image

    trajectory = [] #[xt.clone().detach().cpu()]
    trajectory_denoised = [] #[xt.clone().detach().cpu()]

    for _ in range(initial_step):
        empty_sample = torch.zeros_like(xt).to(device)
        trajectory.append(empty_sample)
        trajectory_denoised.append(empty_sample)
    
    trajectory.append(xt)

    for i in range(initial_step, len(scheduler.timesteps)):
        t = scheduler.timesteps[i]
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
        x_0_hat = (
            scheduler_output.pred_original_sample
        )
        trajectory.append(xt.clone().detach().cpu())
        trajectory_denoised.append(x_0_hat.clone().detach().cpu())
        # if t != scheduler.timesteps[-1]:
        #     trajectory.append(xt.clone().detach().cpu())
        #     # Compute the predicted denoised image at this step
        #     x_0_hat = predict_denoised_image(image_pipe, xt, t, scheduler, device)
        #     trajectory_denoised.append(x_0_hat.clone().detach().cpu())
        # save tensors each 10 steps and in the last

        # if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        #     trajectory.append(xt.clone().detach().cpu())

    # save trajectories in obs' dictionary
    obs["trajectory"] = trajectory
    obs["trajectory_denoised"] = trajectory_denoised

    # now we will release the VRAM memory deleting the variable bounded to the VRAM and use flush()
    del xt
    del noise_pred
    del model_input
    del scheduler_output
    flush()

    return obs