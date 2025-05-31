# data_module.py

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cnn_utils import load_signs_dataset, load_happy_dataset, convert_to_one_hot


class DataModule:
    """
    Loads raw data (NumPy), normalizes, splits into train/val/test, and—
    if use_tf_dataset=True—converts each split into a tf.data.Dataset with batching.

    Public interface:
      - get_train_data()
      - get_val_data()
      - get_test_data()

    Each of those returns:
      * If use_tf_dataset == False: a tuple (X_split, Y_split) of NumPy arrays.
      * If use_tf_dataset == True: a single tf.data.Dataset object of (x,y) pairs.
    """

    def __init__(
        self,
        dataset: str,
        y_type: str = "one_hot_key",
        val_split: float = 0.1,
        random_state: int = 42,
        batch_size: int = 64,
        shuffle_buffer_size: int = 1000,
        use_tf_dataset: bool = False,
    ):
        """
        Args:
          dataset: either "signs" or "face"
          y_type: "one_hot_key" (one‐hot), "one_val" (sparse integer), or "binary_val"
          val_split: fraction of training set to use for validation
          random_state: seed for reproducibility when splitting
          batch_size: batch size for tf.data.Dataset if use_tf_dataset=True
          shuffle_buffer_size: buffer size for shuffling the train dataset
          use_tf_dataset: if True, return tf.data.Dataset objects; if False, return NumPy arrays.
        """
        self.dataset = dataset.lower()
        self.y_type = y_type
        self.val_split = val_split
        self.random_state = random_state

        # For tf.data.Dataset creation
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.use_tf_dataset = use_tf_dataset

        # Internal holders for raw NumPy splits
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None
        self.classes = None

        # Internal holders for tf.data.Dataset splits (only if use_tf_dataset=True)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """
        1) Loads raw data (NumPy) via load_signs_dataset() or load_happy_dataset().
        2) Normalizes X to [0,1].
        3) Converts Y based on y_type.
        4) Splits off validation from training.
        5) If use_tf_dataset=True, creates and stores batched tf.data.Datasets.
        """

        # --- 1) Load raw data ---
        if self.dataset == "signs":
            X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
        elif self.dataset == "face":
            X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()
        else:
            raise ValueError("`dataset` must be 'signs' or 'face'.")

        # --- 2) Normalize inputs to [0, 1] as float32 ---
        X_train = X_train_orig.astype(np.float32) / 255.0
        X_test = X_test_orig.astype(np.float32) / 255.0

        # --- 3) Prepare labels based on y_type ---
        if self.y_type == "one_hot_key":
            num_classes = 6
            Y_train = convert_to_one_hot(Y_train_orig, num_classes).T  # shape (m, num_classes)
            Y_test = convert_to_one_hot(Y_test_orig, num_classes).T
        elif self.y_type == "one_val":
            Y_train = Y_train_orig.reshape(-1).astype(np.int32)  # (m,)
            Y_test = Y_test_orig.reshape(-1).astype(np.int32)
        elif self.y_type == "binary_val":
            Y_train = Y_train_orig.T.astype(np.float32)  # (m,)
            Y_test = Y_test_orig.T.astype(np.float32)
        else:
            raise ValueError("`y_type` must be 'one_hot_key', 'one_val', or 'binary_val'.")

        # --- 4) Split into train/validation (NumPy) ---
        X_train_final, X_val, Y_train_final, Y_val = train_test_split(
            X_train, Y_train, test_size=self.val_split, random_state=self.random_state
        )

        # Store NumPy splits
        self.X_train, self.Y_train = X_train_final, Y_train_final
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test
        self.classes = classes

        # --- 5) If requested, build tf.data.Dataset versions ---
        if self.use_tf_dataset:
            # 5a) Training dataset: shuffle + batch
            train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train))
            train_ds = train_ds.shuffle(buffer_size=self.shuffle_buffer_size)
            train_ds = train_ds.batch(self.batch_size)

            # 5b) Validation dataset: batch only
            val_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.Y_val))
            val_ds = val_ds.batch(self.batch_size)

            # 5c) Test dataset: batch only
            test_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.Y_test))
            test_ds = test_ds.batch(self.batch_size)

            # Store the tf.data.Dataset objects
            self.train_dataset = train_ds
            self.val_dataset = val_ds
            self.test_dataset = test_ds

    def get_train_data(self):
        """
        Returns either:
          * (X_train, Y_train) as NumPy arrays, if use_tf_dataset==False
          * train_dataset as a single tf.data.Dataset, if use_tf_dataset==True
        """
        if self.X_train is None and self.use_tf_dataset is False:
            raise RuntimeError("Call setup() before get_train_data()")

        if self.use_tf_dataset:
            if self.train_dataset is None:
                raise RuntimeError("Call setup() before get_train_data()")
            return self.train_dataset
        else:
            return self.X_train, self.Y_train

    def get_val_data(self):
        """
        Returns either:
          * (X_val, Y_val) as NumPy arrays, if use_tf_dataset==False
          * val_dataset as a single tf.data.Dataset, if use_tf_dataset==True
        """
        if self.X_val is None and self.use_tf_dataset is False:
            raise RuntimeError("Call setup() before get_val_data()")

        if self.use_tf_dataset:
            if self.val_dataset is None:
                raise RuntimeError("Call setup() before get_val_data()")
            return self.val_dataset
        else:
            return self.X_val, self.Y_val

    def get_test_data(self):
        """
        Returns either:
          * (X_test, Y_test) as NumPy arrays, if use_tf_dataset==False
          * test_dataset as a single tf.data.Dataset, if use_tf_dataset==True
        """
        if self.X_test is None and self.use_tf_dataset is False:
            raise RuntimeError("Call setup() before get_test_data()")

        if self.use_tf_dataset:
            if self.test_dataset is None:
                raise RuntimeError("Call setup() before get_test_data()")
            return self.test_dataset
        else:
            return self.X_test, self.Y_test
