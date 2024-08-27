"""Eval script for obtaining samples and rewards for a given ckpt"""

import argparse
import logging
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
# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Define hyparparameters--------------------------------------------------------
# Using argparse to define hyperparameters
parser = argparse.ArgumentParser(
    description="Sample from google/ddpm-celebahq-256 ckpt and compute rewards."
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=25,
)
parser.add_argument(
    "--num_inference_timesteps",
    type=int,
    default=40,
)
parser.add_argument(
    "--task",
    type=Task,
    choices=list(Task),
    default=Task.LAION,
)
parser.add_argument(
    "--output_path",
    type=str,
    default=".",
)
parser.add_argument(
    "--metadata_path",
    type=str,
    default="./metadata.csv",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--ckpt_from_wandb",
    type=str,
    default=None,
    help="Provide the path from W&B model artifact (e.g. alcazar90/ddpo-compressibility-ddpm-celebahq256/Task.COMPRESSIBILITY-generous-deluge-8:v17)",
)
parser.add_argument(
    "--num_batches",
    type=int,
    default=2,
)
parser.add_argument(
    "--initial_steps",
    type=str,
    default="[0]",
    help="Steps to start from"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
# threshold and punishment prameter for under30_old and over50_old rewards
parser.add_argument(
    "--threshold",
    type=float,
    default=0.6,
)
parser.add_argument(
    "--punishment",
    type=float,
    default=-1.0,
),
parser.add_argument(
    "--eval_seed",
    type=str,
    default="620",
    help="Comma-separated list of seeds for initializing random number generation for reproducibility.",
)


# parse the arguments
args = parser.parse_args()
num_samples = args.num_samples
num_inference_timesteps = args.num_inference_timesteps
task = args.task
metadata_path = args.metadata_path
ckpt_path = args.ckpt_path
ckpt_from_wandb = args.ckpt_from_wandb
num_batches = args.num_batches
initial_steps = ast.literal_eval(args.initial_steps)
output_path = args.output_path
device = args.device
threshold = args.threshold
punishment = args.punishment
eval_seeds = list(map(int, args.eval_seed.split(',')))


# Verify if file and folder exists for read and write
# ------------------------------------------------------------------------------
# Check if the metadata file exists
## if not os.path.exists(metadata_path):
##     raise FileNotFoundError("metadata.csv file not found in %s", metadata_path)

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

# Check if ckpt_path and ckpt_from_wandb are both provided
if ckpt_path is not None and ckpt_from_wandb is not None:
    raise ValueError(
        "Both ckpt_path and ckpt_from_wandb cannot be provided. You must choose only a single checkpoint to load."
    )

# Read metadata file
# ------------------------------------------------------------------------------
## metadata = pd.read_csv(metadata_path)

# Download google/ddpm-celebahq-256 image pipeline and scheduler & load ckpt
# ------------------------------------------------------------------------------
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)


# Create new scheduler and set num inference steps
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=num_inference_timesteps)

# Load a ckpt if ckpt_path or ckpt_from_wandb is provided
if ckpt_path is not None or ckpt_from_wandb is not None:
    if ckpt_from_wandb is not None:
        logging.info("Connect to wandb and download the ckpt")
        api = wandb.Api()
        artifact = api.artifact(ckpt_from_wandb)
        artifact_name = ckpt_from_wandb.split("/")[-1]
        # Download the artifact in the current dir
        ckpt_path = artifact.download(".")
        ckpt_path = (
            os.path.join(".", os.path.basename(artifact_name)).split(":")[0]
            + "-ckpt.pth"
        )
        ckpt = torch.load(ckpt_path)
        logging.info("Loading ckpt from %s", ckpt_path)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        logging.info("Loading ckpt from %s", ckpt_path)
    # Load the model state dict
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    logging.info("Ckpt loaded successfully!")

# Set the model to eval mode
image_pipe.unet.eval()

# Download and initialize the reward function
if task == Task.LAION:
    reward_fn = aesthetic_score(
        device=device,
    )
elif task == Task.UNDER30:
    reward_fn = under30_old(
        threshold=threshold,
        punishment=punishment,
        device=device,
    )
elif task == Task.OVER50:
    reward_fn = over50_old(
        threshold=threshold,
        punishment=punishment,
        device=device,
    )
elif task == Task.COMPRESSIBILITY:
    reward_fn = jpeg_compressibility(
        device=device,
    )
elif task == Task.INCOMPRESSIBILITY:
    reward_fn = jpeg_incompressibility(
        device=device,
    )
elif task == Task.MULTITASK:
    reward_fn =[aesthetic_score(device=device),
                jpeg_compressibility(device=device), 
                jpeg_incompressibility(device=device)]


# Running the sampling process, compute metrics and save the results
# ------------------------------------------------------------------------------
# need_denoised_images = True  # Flag to determine when to compute denoised images
count = 0
seeds = []
# for seed in metadata.loc[:, "random_seed"]:
for seed in eval_seeds:
    logging.info("Starting sampling process #%s", count + 1)

    # check if we have reached the number of batches
    if count >= num_batches:
        logging.info("Reached the number of batches. Exiting...")
        break

    # get one random seed
    rnd_seed = seed
    seeds.append(rnd_seed)

    # sample # num_saples from ddpm-celebahq-256 with current rnd_seed
    initial_num_samples = 1
    logging.info(
        "Get #%s from the model with random seed (and key): %s", initial_num_samples, rnd_seed
    )
    data = sample_denoised_images_from_celebahq(
        initial_num_samples, scheduler, image_pipe, device, random_seed=rnd_seed
    )
    # # Optionally predict and store denoised images
    # if need_denoised_images:  # This flag determines when to compute denoised images
    #     logging.info("Predicting and storing denoised images.")
    #     # Assuming image_pipe, scheduler are set up, and images are loaded
    #     denoised_images = predict_and_store_denoised_images_in_batches(data['trajectory'], image_pipe, scheduler, device='cuda')

    #     data['trajectory_noiseless'] = denoised_images
    # # compute rewards
    logging.info("Computing rewards")
    rewards = []
    for xt in data["final_images"]:
        if  task == Task.MULTITASK:
            rewards.append([reward(xt.to(device)).cpu() for reward in reward_fn])
        else:
            rewards.append(reward_fn(xt.to(device)).cpu())
    data["rewards"] = torch.stack(rewards).view(-1).tolist()
    logging.info(
        "Rewards size %s : ",
        len(data["rewards"])
    )
    logging.info("Rewards computed successfully!")

    # save the picle file
    pickle_file_path = os.path.join(output_path, f"id_batch_{rnd_seed}.pkl")
    logging.info("Saving pickle file to %s", pickle_file_path)
    with open(pickle_file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info("Succesfully saved!")
    
    #Sampling from intermediate steps and saving the files

    for step in initial_steps:
        data_intermediate = sample_denoised_images_from_celebahq_intermediate_step(
            num_samples, scheduler, image_pipe, device, step, data["final_images"][-1][0], random_seed=rnd_seed
        )
        logging.info("Computing rewards")
        rewards = []
        for xt in data_intermediate[f"final_images_{step}"]:
            if task == Task.MULTITASK:
                rewards.append([reward(xt.to(device)).cpu() for reward in reward_fn])
            else:
                rewards.append(reward_fn(xt.to(device)).cpu())
        data_intermediate[f"rewards_{step}"] = torch.stack(rewards).view(-1).tolist()
        logging.info(
            "Rewards size %s",
            len(data_intermediate["rewards"])
        )
        logging.info("Rewards computed successfully!")

        # save the picle file
        pickle_file_path = os.path.join(output_path, f"id_batch_{rnd_seed}_initial_step_{step}.pkl")
        logging.info("Saving pickle file to %s", pickle_file_path)
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data_intermediate, f)
        logging.info("Succesfully saved!")

        del data_intermediate
    
    count += 1

logging.info("Create a metadata csv with the current status of the folder...")

# Create metadata file for the output folder
# ------------------------------------------------------------------------------
# Read files in the output folder and filter the files for .pkl files
all_files = os.listdir(output_path)
pkl_files = [file for file in all_files if file.endswith(".pkl")]

# Create a dataframe with the random seeds and the batch name
out_metadata = pd.DataFrame(
    {
        "id": range(len(pkl_files)),
        "task": [task] * len(pkl_files),
        "random_seed": seeds,
        "batch_names": pkl_files,
        "completed": [1.0] * len(pkl_files),
    }
)

# save csv file in the output folder
out_metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)
logging.info("Metadata file for output created and saved successfully!")