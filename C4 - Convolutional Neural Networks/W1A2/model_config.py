from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple
import tensorflow as tf


@dataclass
class ModelConfig:
    """
    Bundles all the information needed to build, train, and save a particular model.
    """

    name: str
    input_shape: Tuple[int, int, int]
    units: int
    save_dir: Path
    build_fn: Callable[..., tf.keras.Model]  # function that builds the model

    def __post_init__(self):
        # Ensure the directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
