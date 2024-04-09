"""Configuration file for DDPO."""

from enum import Enum

EPS = 1e-6


class Task(Enum):
    """The downstream tasks supported to be learn by DDPO."""

    LAION = "aesthetic score"
    UNDER30 = "under30 years old"
    OVER50 = "over50 years old"
    COMPRESSIBILITY = "jpeg compressibility"
    INCOMPRESSIBILITY = "jpeg incompressibility"
