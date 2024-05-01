import torch

def predict_denoised_image(model, x_t, t, scheduler, label=0, device='cuda'):
    """
    Estimate the denoised image from a noisy image at a specific timestep using the model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        x_t (torch.Tensor): The noisy image tensor at timestep t.
        t (int): The current timestep index.
        scheduler (Scheduler): The noise scheduler with alphas_cumprod.
        label (int): The class label for conditional generation.
        device (str): The device to perform computation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The estimated denoised image tensor.
    """
    # Ensure x_t is on the correct device and has the correct dimensions
    x_t = x_t.to(device)
    if x_t.dim() == 3:  # Assuming [B, W, H]
        x_t = x_t.unsqueeze(1)  # Adds a channel dimension [B, C=1, W, H]

    # Convert label to a tensor and ensure it matches the batch size
    y = torch.tensor([label] * x_t.size(0), dtype=torch.long, device=device)

    # Get the necessary parameters for the timestep
    alpha_t = scheduler.alphas_cumprod[t].to(device)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Model evaluation mode and predict noise
    model.eval()
    with torch.no_grad():
        noise_estimate = model(x_t, t, y)

    # Estimate the denoised image
    x_0_hat = (x_t - sqrt_one_minus_alpha_t * noise_estimate) / torch.sqrt(alpha_t)

    return x_0_hat