# run_all.py

from pathlib import Path

import tensorflow as tf
from data_module import DataModule
from model_module import ModelModule
from trainer import Trainer
from cnn_architectures import sequential_conv_model, api_conv_model


def train_sign_seq():
    """
    Trains the simple happyModel on the sign dataset (multi‐class).
    If use_multi_gpu=True, wraps build/compile in MirroredStrategy.
    """
    dm = DataModule(
        dataset="signs",
        y_type="one_hot_key",
        val_split=0.1,
        batch_size=64,
        use_tf_dataset=True,
    )
    dm.setup()
    model_name_: str = "model_sign_seq"
    mm = ModelModule(
        model_name=model_name_,
        build_fn=sequential_conv_model,
        input_shape=(64, 64, 3),
        units=6,
        y_type="one_hot_key",
    )
    mm.build_and_compile(use_lr_schedule=False)

    trainer = Trainer(
        data_module=dm,
        model_module=mm,
        save_dir=Path(f"models/{model_name_}"),
        epochs=100,
        patience=5,
    )
    return trainer.train_and_evaluate()


def train_sign_api():
    """
    Trains the convolutional_model on the sign dataset (multi‐class).
    """
    dm = DataModule(
        dataset="signs",
        y_type="one_hot_key",
        val_split=0.1,
        batch_size=64,
        use_tf_dataset=True,
    )
    dm.setup()
    model_name_: str = "model_sign_api"
    mm = ModelModule(
        model_name=model_name_,
        build_fn=api_conv_model,
        input_shape=(64, 64, 3),
        units=6,
        y_type="one_hot_key",
    )
    mm.build_and_compile(use_lr_schedule=False)

    trainer = Trainer(
        data_module=dm,
        model_module=mm,
        save_dir=Path(f"models/{model_name_}"),
        epochs=100,
        patience=5,
    )
    return trainer.train_and_evaluate()


def train_face_seq():
    """
    Trains the happyModel on the face/happy dataset (binary classification).
    """
    dm = DataModule(
        dataset="face",
        y_type="binary_val",
        val_split=0.1,
        batch_size=64,
        use_tf_dataset=True,
    )
    dm.setup()
    model_name_: str = "model_face_seq"
    mm = ModelModule(
        model_name=model_name_,
        build_fn=sequential_conv_model,
        input_shape=(64, 64, 3),
        units=1,
        y_type="binary_val",
    )
    mm.build_and_compile(use_lr_schedule=False)

    trainer = Trainer(
        data_module=dm,
        model_module=mm,
        save_dir=Path(f"models/{model_name_}"),
        epochs=100,
        patience=5,
    )
    return trainer.train_and_evaluate()


def train_face_api():
    """
    Trains the happyModel on the face/happy dataset (binary classification).
    """
    dm = DataModule(
        dataset="face",
        y_type="binary_val",
        val_split=0.1,
        batch_size=64,
        use_tf_dataset=True,
    )
    dm.setup()
    model_name_: str = "model_face_api"
    mm = ModelModule(
        model_name=model_name_,
        build_fn=api_conv_model,
        input_shape=(64, 64, 3),
        units=1,
        y_type="binary_val",
    )
    mm.build_and_compile(use_lr_schedule=False)

    trainer = Trainer(
        data_module=dm,
        model_module=mm,
        save_dir=Path(f"models/{model_name_}"),
        epochs=100,
        patience=5,
    )
    return trainer.train_and_evaluate()


def main():
    # train_sign_seq()
    # train_sign_api()
    # train_face_seq()
    train_face_api()


if __name__ == "__main__":
    main()
