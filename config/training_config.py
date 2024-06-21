from dataclasses import dataclass
from decouple import config


@dataclass
class TrainingConfig:
    learning_rate: float = float(config('LEARNING_RATE'))
    batch_size: int = int(config('BATCH_SIZE'))
    num_epochs: int = int(config('NUM_EPOCHS'))
    root_dir: str = config('ROOT_DIR_FOR_DATASET')
    num_classes: int = int(config('NUM_CLASSES'))
    results_dir: str = str(config("RESULTS_DIR"))

