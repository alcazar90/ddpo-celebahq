import torch
from tqdm import tqdm

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
