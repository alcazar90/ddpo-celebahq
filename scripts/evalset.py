import argparse
import logging
import os

import pandas as pd
import torch
import wandb
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image

from ddpo.config import DDPMCheckpoint, Task
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)
from ddpo.sampling import sample_from_ddpm
from ddpo.utils import flush

# Set up logging----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Define hyparparameters--------------------------------------------------------


# Add argparser validation argument
def check_ddpm_ckpt(value):
    try:
        # Attempt to match the input value to an enum name
        return DDPMCheckpoint(value).value
    except ValueError:
        # If the value does not match, raise an argparse error
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid DDPM checkpoint. Allowed values are: {[ckpt.value for ckpt in DDPMCheckpoint]}"
        )


def parse_args():
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
        "--ddpm_ckpt",
        type=check_ddpm_ckpt,
        default=DDPMCheckpoint.CELEBAHQ256.value,
        help="The DDPM pretrained model checkpoint from Hugging Face. Allowed values are: "
        + ", ".join([ckpt.value for ckpt in DDPMCheckpoint]),
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
        "--start_from",
        type=int,
        default=0,
        help="The row index to start from in the metadata file.",
    )
    parser.add_argument(
        "--until_to",
        type=int,
        help="Index of the batch to process until",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    return args


args = parse_args()


# Define helper functions
# ------------------------------------------------------------------------------
def save_images(images, output_dir, batch_idx, start_idx):
    """Saves images as PNG files.

    Args:
      images: A tensor of images.
      output_dir: The directory to save the images to.
      batch_idx: The index of the batch (it comes from metadata file).
      start_idx: The starting index for the image filenames.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image in enumerate(images):
        image = (
            ((image.detach().cpu().clip(-1, 1) * 0.5 + 0.5) * 255)
            .to(torch.uint8)
            .numpy()
        )
        image = image.transpose(1, 2, 0)  # CHW -> HWC
        image = Image.fromarray(image)
        image.save(os.path.join(output_dir, f"{batch_idx}_{start_idx + i}.png"))


def save_rewards(rewards, output_dir, batch_idx, num_samples):
    """Saves rewards as a CSV file.

    Args:
      rewards: A tensor of rewards.
      output_dir: The directory to save the rewards to.
      batch_idx: The index of the batch (it comes from metadata file).
      num_samples: The number of samples in the batch.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rwds_df = pd.DataFrame(
        {
            "id": [str(batch_idx) + "_" + str(x) for x in range(num_samples)],
            "reward": rwds.to("cpu").detach().numpy(),
        }
    )
    rwds_df.to_csv(os.path.join(output_dir, f"{batch_idx}.csv"), index=False)


def stack_csv_files(directory):
    """Stacks all CSV files in a directory vertically.

    Args:
        directory: The directory containing the CSV files.

    Returns:
        A pandas DataFrame containing the stacked data.
    """

    all_dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# Verify if file and folder exists for read and write
# ------------------------------------------------------------------------------
# Check if the metadata file exists
if not os.path.exists(args.metadata_path):
    raise FileNotFoundError("metadata.csv file not found in %s", args.metadata_path)

# Check if the output folder exists. If not, create it
if not os.path.exists(args.output_path):
    logging.info("Creating output folder %s", args.output_path)
    os.makedirs(args.output_path)

# Check if ckpt_path and ckpt_from_wandb are both provided
if args.ckpt_path is not None and args.ckpt_from_wandb is not None:
    raise ValueError(
        "Both ckpt_path and ckpt_from_wandb cannot be provided. You must choose only a single checkpoint to load."
    )


# Read metadata file
# ------------------------------------------------------------------------------
metadata = pd.read_csv(args.metadata_path)


# Download google/ddpm-models image pipeline and scheduler & load ckpt
# ------------------------------------------------------------------------------
image_pipe = DDPMPipeline.from_pretrained(args.ddpm_ckpt)
image_pipe.to(args.device)


# Create new scheduler and set num inference steps
scheduler = DDIMScheduler.from_pretrained(args.ddpm_ckpt)
scheduler.set_timesteps(num_inference_steps=args.num_inference_timesteps)

# Load a ckpt if ckpt_path or ckpt_from_wandb is provided
if args.ckpt_path is not None or args.ckpt_from_wandb is not None:
    if args.ckpt_from_wandb is not None:
        logging.info("Connect to wandb and download the ckpt")
        api = wandb.Api()
        artifact = api.artifact(args.ckpt_from_wandb)
        artifact_name = args.ckpt_from_wandb.split("/")[-1]
        # Download the artifact in the current dir
        ckpt_path = artifact.download(".")
        ckpt_path = (
            os.path.join(".", os.path.basename(artifact_name)).split(":")[0]
            + "-ckpt.pth"
        )
        ckpt = torch.load(ckpt_path)
        logging.info("Loading ckpt from %s", ckpt_path)
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        logging.info("Loading ckpt from %s", args.ckpt_path)
    # Load the model state dict
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    flush()
    logging.info("Ckpt loaded successfully!")

# Set the model to eval mode
image_pipe.unet.eval()

# Download and initialize the reward function
if args.task == Task.LAION:
    reward_fn = aesthetic_score(
        device=args.device,
    )
elif args.task == Task.UNDER30:
    reward_fn = under30_old()
elif args.task == Task.OVER50:
    reward_fn = over50_old(
        device=args.device,
    )
elif args.task == Task.COMPRESSIBILITY:
    reward_fn = jpeg_compressibility(
        device=args.device,
    )
elif args.task == Task.INCOMPRESSIBILITY:
    reward_fn = jpeg_incompressibility(
        device=args.device,
    )


# Running the sampling process, compute metrics and save the results
# ------------------------------------------------------------------------------
for row_idx in range(args.start_from, min(metadata.shape[0], args.until_to + 1)):
    batch_idx = metadata["id"][row_idx]
    random_seed = metadata["random_seed"][row_idx]

    logging.info("Processing batch %d - Using seed %d", batch_idx, random_seed)

    # Get the batch images
    batch_images = sample_from_ddpm(
        num_samples=args.num_samples,
        scheduler=scheduler,
        image_pipe=image_pipe,
        device=args.device,
        random_seed=random_seed,
    )

    logging.info("Computing rewards")
    # Compute rewards and save rewards into a CSV
    rwds = reward_fn(batch_images)

    save_rewards(rwds, f"{args.output_path}/rewards", batch_idx, args.num_samples)
    logging.info("Rewards saved successfully!")

    # Save images in png format
    save_images(batch_images, f"{args.output_path}/samples", batch_idx, start_idx=0)
    logging.info("Images saved successfully!")


# At the end stack the reward CSV and save in the same folder that png images
logging.info("Stacking rewards CSV files")
stacked_df = stack_csv_files(f"{args.output_path}/rewards")
stacked_df.to_csv(f"{args.output_path}/rewards.csv", index=False)
logging.info("Reward CSV files stacked and saved successfully!")
