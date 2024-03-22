from enum import Enum

class Task(Enum):
    # The downstream tasks to be learn by DDPO
    LAION = "aesthetic score - laion aesthetic"
    UNDER30 = "under 30 years old - vit age classifier"
    OVER50 = "over 50 years old - vit age classifier"