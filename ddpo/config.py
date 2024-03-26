from enum import Enum

class Task(Enum):
    # The downstream tasks to be learn by DDPO
    LAION = "aesthetic score"
    UNDER30 = "under30 years old"
    OVER50 = "over50 years old"
    COMPRESSIBILITY = "jpeg compressibility"
    INCOMPRESSIBILITY = "jpeg incompressibility"
