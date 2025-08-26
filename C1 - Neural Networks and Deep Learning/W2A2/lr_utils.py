import numpy as np
import h5py
from pathlib import Path
import os


def get_root_folder():
    """
    Locate the root project folder by searching for a marker file (e.g., .projectroot).
    If not found, assume the current working directory is the root.
    """
    # Start from the current working directory (for Jupyter Notebooks)
    current_path = Path(os.getcwd()).resolve()

    # If running as a script, start from the script's directory
    if "__file__" in globals():
        current_path = Path(__file__).resolve().parent

    # Traverse up the directory tree to find the root folder
    for parent in current_path.parents:
        if (parent / ".projectroot").exists():  # Check for a marker file
            return parent
    return current_path  # Fallback to the current directory


def get_folder_path(folder_name):
    """
    Get the absolute path of a folder in the root project folder.
    """
    root_folder = get_root_folder()
    folder_path = root_folder / folder_name
    return folder_path.resolve()


def load_dataset():
    resources_dir: Path = get_folder_path("_resources_deep_learning") / "c1" / "W2_A2" / "datasets"
    dataset_path: Path = resources_dir / "train_catvnoncat.h5"
    if dataset_path.exists():
        print("it exists")
    else:
        print("it doesnt exist")
    print(f"Trying to open: {dataset_path}")  # ADD THIS LINE
    train_dataset = h5py.File(dataset_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    dataset_path: Path = resources_dir / "test_catvnoncat.h5"
    test_dataset = h5py.File(dataset_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
