import numpy as np
import h5py
from pathlib import Path
from common import get_folder_path, get_root_folder


def get_resource_path(dataset_path: str) -> Path:
    helper_dir = Path(__file__).parent
    rel_path = helper_dir / dataset_path
    abs_path = rel_path.resolve()
    return abs_path


def load_dataset():
    resources_dir: Path = get_folder_path("_resources_deep_learning")
    dataset_path: Path = resources_dir / "/W2A2" / "datasets" / "train_catvnoncat.h5"
    train_dataset = h5py.File(str(dataset_path), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    dataset_path: Path = resources_dir / "W2A2" / "datasets" / "test_catvnoncat.h5"
    test_dataset = h5py.File(str(dataset_path), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
