import torch
from tqdm import tqdm
from ddpo.utils import flush
import pickle
import gc  # For garbage collection
import pandas as pd
import wandb
import os
import logging
from diffusers import DDIMScheduler, DDPMPipeline


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
def sample_denoised_data_with_initial_state(
    scheduler,
    image_pipe,
    device,
    initial_states,
    start_step,
    random_seed=None
):
    """
    Sample a batch of trajectories using a specified scheduler and image pipeline, starting from given initial states at a specific step.

    Args:
        scheduler (DDIMScheduler): The scheduler object that controls the sampling process.
        image_pipe (ImagePipeline): The image pipeline object used for processing images.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform computations.
        initial_states (list of tensors): The initial states (images) from which to start the denoising process.
        start_step (int): The index of the timestep from which to start the denoising process.
        random_seed (int, optional): The random seed for reproducibility.

    Returns:
        dict: Contains "trajectory" and "trajectory_denoised", tensors containing the trajectories of the entire batch (T, B, C, H, W).
    """
    obs = {}
    if random_seed is not None:
        obs["seed"] = random_seed
        torch.manual_seed(random_seed)

    xt = initial_states[start_step].to(device)

    trajectory = [xt.clone().detach().cpu()]
    trajectory_denoised = [xt.clone().detach().cpu()]

    # Start denoising from the specified step
    for i in range(start_step, len(scheduler.timesteps)):
        t = scheduler.timesteps[i]

        # Scale input based on the timestep
        model_input = scheduler.scale_model_input(xt, t)

        with torch.no_grad():
            # Get the noise prediction
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Compute the update sample regarding the scheduler
        scheduler_output = scheduler.step(noise_pred, t, xt)

        # Update x
        xt = scheduler_output.prev_sample
        x_0_hat = scheduler_output.pred_original_sample

        trajectory.append(xt.clone().detach().cpu())
        trajectory_denoised.append(x_0_hat.clone().detach().cpu())

    # Save trajectories in obs' dictionary
    obs["trajectory"] = trajectory
    obs["trajectory_denoised"] = trajectory_denoised

    return obs



def process_model(image_pipe, scheduler, device, num_samples, seed, reward_fn, output_path, ckpt_path=None, ckpt_from_wandb=None):
    # Load the model checkpoint
    ckpt_pth = load_model(ckpt_path, ckpt_from_wandb)
    if ckpt_pth is None:
        logging.error("Failed to load model checkpoint.")
        return

    ckpt = torch.load(ckpt_pth)  # Adjust based on actual needs for loading
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    image_pipe.unet.eval()

    # Proceed with sampling
    data = sample_denoised_data_from_celebahq(
        num_samples, scheduler, image_pipe, device, random_seed=seed
    )
    rewards = []
    for xt in data["trajectory"]:
        rewards.append(reward_fn(xt.to(device)).cpu())
    data["rewards"] = torch.stack(rewards).view(-1).tolist()
    logging.info(
        "Rewards size %s | First 5 rewards: %s",
        len(data["rewards"]),
        data["rewards"][:5],
    )
    logging.info("Rewards computed successfully!")

    # Save the pickle file
    pickle_file_path = os.path.join(output_path, f"id_batch_{seed}.pkl")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info("Data saved successfully at %s", pickle_file_path)

    return data
    # # Clean up
    # del model
    # torch.cuda.empty_cache()
    # gc.collect()

def process_model_data_injection(image_pipe, scheduler, device, seed, reward_fn, output_path, initial_data, start_steps, ckpt_path=None, ckpt_from_wandb=None):
    # Load the model checkpoint
    ckpt_pth = load_model(ckpt_path, ckpt_from_wandb)
    if ckpt_pth is None:
        logging.error("Failed to load model checkpoint.")
        return

    ckpt = torch.load(ckpt_pth)  # Load the checkpoint
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    image_pipe.unet.eval()

    # Iterate over the list of start steps
    for start_step in start_steps:
        logging.info(f"Starting denoising from step index {start_step}")

        # Proceed with sampling starting from the specified step
        data = sample_denoised_data_with_initial_state(
            scheduler,
            image_pipe,
            device,
            initial_data["trajectory"],  # Use the provided initial trajectories
            start_step,
            random_seed=seed
        )

        rewards = []
        for xt in data["trajectory"]:
            rewards.append(reward_fn(xt.to(device)).cpu())
        # Check if the rewards list has at least five elements, if not, show as many as available
        last_rewards = data["rewards"][-5:] if len(data["rewards"]) >= 5 else data["rewards"]
        logging.info(
            "Rewards computed successfully for start step {} | Rewards size: {} | Last 5 rewards: {}".format(
                start_step, len(data["rewards"]), last_rewards))


        # Save the pickle file with a name reflecting the start step and seed
        pickle_file_path = os.path.join(output_path, f"id_{start_step}_batch_{seed}.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data, f)
        logging.info("Data saved successfully at %s", pickle_file_path)


def load_model(ckpt_path=None, ckpt_from_wandb=None):
    if ckpt_from_wandb:
        logging.info("Connect to wandb and download the ckpt")
        api = wandb.Api()
        artifact = api.artifact(ckpt_from_wandb)
        artifact_name = ckpt_from_wandb.split("/")[-1]
        # Download the artifact in the current directory
        ckpt_path = artifact.download(".")
        ckpt_path = os.path.join(".", os.path.basename(artifact_name)).split(":")[0] + "-ckpt.pth"

    if ckpt_path is not None:
        logging.info("Loading checkpoint from %s", ckpt_path)
        ckpt = torch.load(ckpt_path)
        logging.info("Checkpoint loaded successfully!")
        return ckpt
    else:
        logging.error("No checkpoint path provided.")
        return None
