# model_module.py

import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class ModelModule:
    """
    Wraps a Keras model architecture and compilation step.
    """

    def __init__(
        self, model_name: str, build_fn, input_shape, units: int, y_type: str = "one_hot_key"
    ):
        """
        Args:
            model_name: descriptive string (e.g. "simple_happy", "conv_sign").
            build_fn: a function that takes (input_shape, is_binary, units) → tf.keras.Model.
            input_shape: tuple of ints, e.g. (64, 64, 3).
            units: number of output neurons (1 if binary, else num_classes).
            y_type: "one_hot_key", "one_val", or "binary_val".  Used to pick loss.
        """
        self.model_name = model_name
        self.build_fn = build_fn
        self.input_shape = input_shape
        self.units = units
        self.y_type = y_type

        self.model = None  # Will be set once we call build_and_compile()

    def build_and_compile(self, use_lr_schedule: bool = False):
        """
        Instantiates the Keras architecture (calls build_fn) and compiles it.
        If use_lr_schedule=True, uses a simple ExponentialDecay schedule for Adam.
        """
        is_binary = self.y_type == "binary_val"

        # 1) Build (architecture)
        self.model = self.build_fn(self.input_shape, is_binary=is_binary, units=self.units)

        # 2) Decide on loss
        if self.y_type == "one_hot_key":
            loss_fn = "categorical_crossentropy"
        elif self.y_type == "one_val":
            loss_fn = "sparse_categorical_crossentropy"
        else:  # binary
            loss_fn = "binary_crossentropy"

        # 3) Choose optimizer
        if use_lr_schedule:
            lr_schedule = ExponentialDecay(
                initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9, staircase=True
            )
            optimizer = Adam(learning_rate=lr_schedule)
        else:
            optimizer = Adam(learning_rate=1e-3)

        # 4) Compile
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
        return self.model

    def summary(self):
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Call build_and_compile() first.")
        self.model.summary()
