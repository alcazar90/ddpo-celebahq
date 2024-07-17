"""Configuration file for DDPO."""

from enum import Enum

# Constants
EPS = 1e-6  # epsilon for numerical stability (avoid division by zero)


class Task(Enum):
    """The downstream tasks supported to be learn by DDPO."""

    LAION = "aesthetic score"
    UNDER30 = "under30 years old"
    OVER50 = "over50 years old"
    COMPRESSIBILITY = "jpeg compressibility"
    INCOMPRESSIBILITY = "jpeg incompressibility"


class DDPMCheckpoint(Enum):
    """The checkpoints for the DDPM model in Hugging Face."""

    CELEBAHQ256 = "google/ddpm-celebahq-256"
    CHURCH256 = "google/ddpm-church-256"
    BEDROOM256 = "google/ddpm-bedroom-256"
