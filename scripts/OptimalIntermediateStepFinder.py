import torch
import numpy as np
import logging
from tqdm import tqdm

import argparse
import os
import pickle
import ast

import pandas as pd
import torch
import wandb
from diffusers import DDIMScheduler, DDPMPipeline

from ddpo.config import Task
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)
from ddpo.sampling import sample_data_from_celebahq, sample_from_ddpm_celebahq
from trayectory_metrics import sample_denoised_images_from_celebahq_intermediate_step
from trayectory_metrics import sample_denoised_images_from_celebahq


class OptimalIntermediateStepFinder:
    def __init__(
        self,
        initial_image,
        scheduler,
        image_pipe,
        reward_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_samples=25,
        initial_steps=None,
        beta=1.0,  # Weighting factor for combining rewards and distance metrics
        random_seed=None,
        task = None
    ):
        """
        Initialize the class with necessary parameters.
        """
        self.initial_image = initial_image.to(device)
        self.scheduler = scheduler
        self.image_pipe = image_pipe
        self.reward_fn = reward_fn
        self.device = device
        self.num_samples = num_samples
        self.initial_steps = initial_steps if initial_steps is not None else [0]
        self.beta = beta
        self.random_seed = random_seed
        self.task = task

        # Data storage
        self.mean_rewards_per_step = {}
        self.mean_distances_per_step = {}
        self.optimal_step = None

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def sample_from_intermediate_steps(self):
        """
        Sample images from various intermediate steps starting from the initial image.
        """
        logging.info("Sampling from intermediate steps...")
        for step in self.initial_steps:
            logging.info(f"Sampling from step {step}")
            # Use the provided sampling function
            data_intermediate = sample_denoised_images_from_celebahq_intermediate_step(
                num_samples=self.num_samples,
                scheduler=self.scheduler,
                image_pipe=self.image_pipe,
                device=self.device,
                initial_step=step,
                final_image_original=self.initial_image,
                random_seed=self.random_seed
            )
            # Extract the images
            images_key = f"final_images_{step}"
            sampled_images = data_intermediate[images_key]

             # Compute rewards for these images
            rewards = []
            for img in sampled_images:
                img_tensor = img.to(self.device)
                if self.task == 'MULTITASK':
                    reward_values = [reward(img_tensor).cpu() for reward in self.reward_fn]
                    reward_tensor = torch.cat(reward_values, dim=0)  # Tensor of shape [num_rewards]
                    rewards.append(reward_tensor)
                else:
                    reward_value = self.reward_fn(img_tensor).cpu()
                    rewards.append(reward_value)
            # Stack rewards
            rewards_tensor = torch.stack(rewards)  # Shape: [num_samples, num_rewards]
            # Compute mean reward per reward component
            mean_rewards = torch.mean(rewards_tensor, dim=0).numpy()  # Shape: [num_rewards]
            self.mean_rewards_per_step[step] = mean_rewards

            # Compute distances for these images
            distances = []
            for img in sampled_images:
                distance = self._compute_cosine_distance(img.to(self.device), self.initial_image)
                distances.append(distance)
            mean_distance = np.mean(distances)
            self.mean_distances_per_step[step] = mean_distance

            # Clean up
            del data_intermediate
            del sampled_images
            torch.cuda.empty_cache()

        logging.info("Sampling from intermediate steps completed.")

    def _prepare_image(self, image):
        """
        Prepare image tensor for reward computation.
        """
        # Assuming images are tensors in the range [-1, 1], convert to [0, 1]
        image = (image + 1) / 2  # Normalize to [0, 1]
        return image

    def _compute_cosine_distance(self, image1, image2):
        """
        Compute the Cosine Distance between two images.
        """
        # Flatten the images
        image1_flat = image1.flatten()
        image2_flat = image2.flatten()

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(image1_flat, image2_flat, dim=0, eps=1e-8)

        # Compute cosine distance
        cos_dist = 1 - cos_sim.item()

        return cos_dist

    def _normalize(self, values):
        """
        Normalize a list of values to range [0, 1].
        """
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val == 0:
            return [0 for _ in values]
        return [(v - min_val) / (max_val - min_val) for v in values]

    def determine_optimal_step(self):
        """
        Determine the optimal intermediate step based on rewards and distance metrics.
        """
        logging.info("Determining optimal intermediate step...")
        steps = self.initial_steps
        mean_rewards = [self.mean_rewards_per_step[step] for step in steps]
        mean_distances = [self.mean_distances_per_step[step] for step in steps]

        # Normalize rewards and distances
        norm_rewards = self._normalize(mean_rewards)
        norm_distances = self._normalize(mean_distances)

        # Compute combined metric
        combined_metric = [r - self.beta * d for r, d in zip(norm_rewards, norm_distances)]

        # Identify the step with the maximum combined metric
        max_index = np.argmax(combined_metric)
        self.optimal_step = steps[max_index-1] # get the previous step so the policy that is favores is acted upon

        logging.info(f"Optimal intermediate step determined: {self.optimal_step}, with best acted step: {steps[max_index]}")

    def run(self):
        """
        Run the entire process.
        """
        self.sample_from_intermediate_steps()
        self.determine_optimal_step()
        return self.optimal_step

# Include your sampling function
@torch.no_grad()
def sample_denoised_images_from_celebahq_intermediate_step(
    num_samples,
    scheduler,
    image_pipe,
    device,
    initial_step,
    final_image_original,
    random_seed=None,
):
    """Sample images from a specified intermediate step."""
    initial_step = torch.tensor(initial_step).to(device).item()
    obs = {}

    if random_seed is not None:
        obs["seed"] = random_seed
        torch.manual_seed(random_seed)

    # Initialize a batch of random noise
    xt = torch.randn(num_samples, 3, 256, 256).to(device)

    # Generate the noisy images
    for i in range(num_samples):
        noisy_image = add_noise_to_image(final_image_original, int(scheduler.timesteps[initial_step]), scheduler)
        xt[i] = noisy_image

    final_image = []

    for _ in range(initial_step):
        # Placeholder for initial steps (not used in sampling)
        pass

    for i in range(initial_step, len(scheduler.timesteps)):
        t = scheduler.timesteps[i]
        # Scale input based on the timestep
        model_input = scheduler.scale_model_input(xt, timestep=t)

        # Get the noise prediction
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t).sample

        # Compute the updated sample according to the scheduler
        scheduler_output = scheduler.step(noise_pred, t, xt)

        # Update x
        xt = scheduler_output.prev_sample
        x_0_hat = scheduler_output.pred_original_sample
        if t == scheduler.timesteps[-1]:
            final_image.append(x_0_hat.clone().detach().cpu())

    # Save the final images
    obs[f"final_images_{initial_step}"] = final_image

    # Clean up
    del xt
    del noise_pred
    del model_input
    del scheduler_output
    torch.cuda.empty_cache()

    return obs

# Function to add noise to an image at a given timestep
def add_noise_to_image(image, timestep, scheduler):
    device = image.device
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(device)

    # Get the variance schedule
    alpha_prod = scheduler.alphas_cumprod[timestep]
    beta_prod = 1 - alpha_prod

    # Sample noise
    noise = torch.randn_like(image).to(device)

    # Add noise to the image
    noisy_image = alpha_prod.sqrt() * image + beta_prod.sqrt() * noise

    return noisy_image


