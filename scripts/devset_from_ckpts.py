"""Sample the devset using different checkpoints associated in W&B run and
compute data associated with the devset."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import wandb
from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
from tqdm.auto import tqdm

from ddpo.config import Task
from ddpo.ddpo import evaluation_loop
from ddpo.rewards import (
    aesthetic_score,
    jpeg_compressibility,
    jpeg_incompressibility,
    over50_old,
    under30_old,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Define config parameters -----------------------------------------------------
parser = argparse.ArgumentParser(
    description="Sample the devset using different checkpoints associated in a W&B's run."
)
parser.add_argument(
    "--project_path",
    type=str,
    default="alcazar90/ddpo-compressibility-ddpm-celebahq256",
    help="W&B project path: user/project-name.",
)
parser.add_argument(
    "--artifact_name",
    type=str,
    default="Task.COMPRESSIBILITY-generous-deluge-8",
    help="W&B artifact name.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=".",
    help="Output directory to save the data.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run the inference.",
)
parser.add_argument(
    "--num_inference_timesteps",
    type=int,
    default=40,
    help="Number of inference timesteps using for the DDIMScheduler to sample.",
)
parser.add_argument(
    "--num_eval_samples",
    type=int,
    default=64,
    help="Number of samples to evaluate the reward function.",
)
parser.add_argument(
    "--eval_random_seed",
    type=int,
    default=650,
    help="Random seed to sample the devset.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.7,
    help="Threshold to use in the reward function UNDER30 and OVER50.",
)
parser.add_argument(
    "--punishment",
    type=float,
    default=0.0,
    help="Punishment to use in the reward function UNDER30 and OVER50.",
)
args = parser.parse_args()

project_path = Path(args.project_path)
artifact_name = args.artifact_name
output_dir = Path(args.output_dir)
device = args.device
num_inference_timesteps = args.num_inference_timesteps
num_eval_samples = args.num_eval_samples
eval_random_seed = args.eval_random_seed
threshold = args.threshold
punishment = args.punishment

# Extract the task from the artifact name
# , e.g. "Task.COMPRESSIBILITY-generous-deluge-8" -> "COMPRESSIBILITY"
task = artifact_name.split(".")[-1].split("-")[0]

# Initialize connection with W&B---------------------------------------------
api = wandb.Api()

# Obtain artifact versions
project_name = Path(project_path)
artifacts = api.artifacts("model", str(project_name / Path(artifact_name)))

# Print all versions of the artifact
logging.info(
    "There are %s checkpoint versions associated to run %s",
    len(artifacts),
    artifact_name,
)

# Initialize diffusion pipeline and scheduler-----------------------------------
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

# Create new scheduler and set num inference steps
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=num_inference_timesteps)

# Instantiate the reward fun according to the run-------------------------------
# Load the reward function based on the task specified in the artifact_name
if task == Task.LAION.name:
    reward_fn = aesthetic_score(device=device)
elif task == Task.UNDER30:
    reward_fn = under30_old(threshold=threshold, punishment=punishment, device=device)
elif task == Task.OVER50.name:
    reward_fn = over50_old(threshold=threshold, punishment=punishment, device=device)
elif task == Task.COMPRESSIBILITY.name:
    reward_fn = jpeg_compressibility(device=device)
elif task == Task.INCOMPRESSIBILITY.name:
    reward_fn = jpeg_incompressibility(device=device)

# Iterate, download ckpt versions, and sample from it---------------------------
for version in tqdm(artifacts):
    ckpt_version = version.version
    version_name = version.name
    # Download the version ckpt
    artifact_version = api.artifact(version)
    download_dir = artifact_version.download(
        output_dir
    )  # download in the output_dir directory
    logging.info("Downloaded %s successfully", version_name)
    logging.info("Initialized the model with %s", version_name)

    # Load ckpt and set the model to eval mode
    ckpt = torch.load(
        str(output_dir / Path(artifact_name)) + "-ckpt.pth"
    )  # follow the naming convention used in train.py (see ckt_path in train.py)
    image_pipe.unet.load_state_dict(ckpt["model_state_dict"])
    image_pipe.unet.eval()

    # Sample the devset using the eval_rnd_seed
    logging.info(
        "Sampling the devset (random seed %s) using %s", eval_random_seed, version_name
    )
    eval_imgs, eval_rdf, eval_logp, _ = evaluation_loop(
        reward_fn,
        scheduler,
        image_pipe,
        device,
        num_samples=num_eval_samples,
        random_seed=eval_random_seed,
    )

    logging.info("Sampling using evaluation_loop run successfully")
    logging.info("Process the data and save it in a dataframe...")
    # change name to the columns in eval_rdf
    # (trajectory's step x num_eval_samples)
    eval_rdf.columns = ["sample_" + str(c + 1) for c in eval_rdf.columns]

    # add metadata in the eval_rdf
    eval_rdf["ckpt_version"] = ckpt_version
    eval_rdf["ckpt_name"] = version_name
    eval_rdf["task"] = task
    eval_rdf["eval_random_seed"] = eval_random_seed
    eval_rdf["best_reward"] = ckpt["best_reward"]

    # change the order of columns
    eval_rdf = eval_rdf[
        ["ckpt_name", "ckpt_version", "task", "eval_random_seed", "best_reward"]
        + list(eval_rdf.columns[:-5])
    ]
    eval_rdf.to_csv(output_dir / f"{version_name}-devset.csv", index=False)

    # save the logp tensor (T, num_eval_samples) as a csv
    eval_logp = pd.DataFrame(eval_logp.numpy())
    eval_logp.columns = ["sample_" + str(c + 1) for c in eval_rdf.columns]
    eval_logp.to_csv(output_dir / f"{version_name}-logp.csv", index=False)

    # save the image tensor
    logging.info("Save the sample torch.tensor")
    torch.save(eval_imgs, output_dir / f"{version_name}-sample-torchtensor.pkl")

    logging.info("Save the samples as png images")
    # TODO: terminar de revisar esto...
    for i, img in enumerate(eval_imgs):
        img = (img.clip(-1, 1) * 0.5 + 0.5) * 255
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img.astype("uint8")
        img = Image.fromarray(img)
        img.save(output_dir / f"{version_name}-sample_{i}.png")

    logging.info("End with version %s", version_name)
