import enum

from .create_dataset import *
from .callbacks import *
from .training import *


class Bert4RecTask(str, enum.Enum):
    NEW = "new"
    EXISTING = "existing"

    def __str__(self):
        return self.value