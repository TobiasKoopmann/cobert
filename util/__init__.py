import enum

from .create_dataset import *
from .callbacks import *
from .training import *


class Bert4RecTask(str, enum.Enum):
    RANKING = "ranking"
    PREDICT = "predict"

    def __str__(self):
        return self.value