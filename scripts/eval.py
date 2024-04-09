"""Eval script for obtaining samples and rewards for a given ckpt"""

import argparse
import logging
import os
import pickle

import pandas as pd
import torch
from diffusers import DDIMScheduler, DDPMPipeline

from ddpo.config import Task
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)
from ddpo.sampling import sample_data_from_celebahq

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
parser = argparse.ArgumentParser(description="DDPO")

parser.add_argument("--num_samples", type=int, default=25)
parser.add_argument("--num_inference_timesteps", type=int, default=40)
parser.add_argument("--task", type=Task, choices=list(Task), default=Task.LAION)
parser.add_argument("--output_path", type=str, default=".")
parser.add_argument("--metadata_path", type=str, default="./metadata.csv")
parser.add_argument("--ckpt_path", type=str, default="./ckpt.pth")
parser.add_argument("--num_batches", type=int, default=2)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
# threshold and punishment prameter for under30_old and over50_old rewards
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--punishment", type=float, default=-1.0)

# parse the arguments
args = parser.parse_args()
num_samples = args.num_samples
num_inference_timesteps = args.num_inference_timesteps
task = args.task
metadata_path = args.metadata_path
ckpt_path = args.ckpt_path
num_batches = args.num_batches
output_path = args.output_path
device = args.device
threshold = args.threshold
punishment = args.punishment


# Verify if file and folder exists for read and write
# ------------------------------------------------------------------------------
# Check if the metadata file exists
if not os.path.exists(metadata_path):
    raise FileNotFoundError("metadata.csv file not found in %s", metadata_path)

# Check if ckpt exists
if not os.path.exists(ckpt_path):
    raise FileNotFoundError("ckpt.pth file not found in %s", ckpt_path)

# Check if the output folder exists. If not, create it
if not os.path.exists(output_path):
    logging.info("Creating output folder %s", output_path)
    os.makedirs(output_path)

# Obtain the folder name from output_path and check if it's equal to celebahq-sample-dataset. If it is, raise an error
folder_name = os.path.basename(output_path)
if folder_name == "celebahq-sample-dataset":
    raise ValueError(
        "Output folder cannot be celebahq-sample-dataset. Please provide a different folder name."
    )

# Read metadata file
# ------------------------------------------------------------------------------
metadata = pd.read_csv(metadata_path)

# Download google/ddpm-celebahq-256 image pipeline and scheduler & load ckpt
# ------------------------------------------------------------------------------
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

# Load ckpt and set the model to eval mode
ckpt = torch.load(ckpt_path)
image_pipe.unet.load_state_dict(ckpt["model_state_dict"])

# Create new scheduler and set num inference steps
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=num_inference_timesteps)

# Download and initialize the reward model
if task == Task.LAION:
    reward_model = aesthetic_score()
elif task == Task.UNDER30:
    reward_model = under30_old(threshold=threshold, punishment=punishment)
elif task == Task.OVER50:
    reward_model = over50_old(threshold=threshold, punishment=punishment)
elif task == Task.COMPRESSIBILITY:
    reward_model = jpeg_compressibility()
elif task == Task.INCOMPRESSIBILITY:
    reward_model = jpeg_incompressibility()


# Running the sampling process, compute metrics and save the results
# ------------------------------------------------------------------------------
count = 0
for seed in metadata.loc[:, "random_seed"]:
    logging.info("Starting sampling process #%s", count + 1)

    # check if we have reached the number of batches
    if count >= num_batches:
        logging.info("Reached the number of batches. Exiting...")
        break

    # get one random seed
    rnd_seed = seed

    # sample # num_saples from ddpm-celebahq-256 with current rnd_seed
    logging.info(
        "Get #%s from the model with random seed (and key): %s", num_samples, rnd_seed
    )

    data = sample_data_from_celebahq(
        num_samples, scheduler, image_pipe, device, random_seed=rnd_seed
    )
    # compute rewards
    logging.info("Computing rewards")
    rewards = []
    for xt in data["trajectory"]:
        rewards.append(reward_model(xt.to(device)).cpu())
    data["rewards"] = torch.stack(rewards).view(-1).tolist()
    logging.info(
        "Rewards size %s | First 5 rewards: %s",
        len(data["rewards"]),
        data["rewards"][:5],
    )
    logging.info("Rewards computed successfully!")

    # save the picle file
    pickle_file_path = os.path.join(output_path, f"id_batch_{rnd_seed}.pkl")
    logging.info("Saving pickle file to %s", pickle_file_path)
    with open(pickle_file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info("Succesfully saved!")
    count += 1
