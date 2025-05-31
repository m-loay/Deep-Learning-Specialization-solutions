# trainer.py

import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import EarlyStopping
import tensorflow as tf  # for isinstance checks


class Trainer:
    """
    Given a DataModule and a ModelModule, runs the training loop,
    evaluates on test data, saves model & plots.
    """

    def __init__(
        self,
        data_module,
        model_module,
        save_dir: Path,
        epochs: int = 100,
        patience: int = 5,
    ):
        """
        Args:
            data_module: an instance of DataModule (already .setup() called).
            model_module: an instance of ModelModule (already .build_and_compile() called).
            save_dir: pathlib.Path where to save model file + plots.
            batch_size, epochs: for model.fit().
            patience: for EarlyStopping on val_loss.
        """
        self.data_module = data_module
        self.model_module = model_module
        self.save_dir = save_dir
        self.batch_size = self.data_module.batch_size
        self.epochs = epochs
        self.patience = patience

        # Create directory if it does not exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(self):
        """
        Runs:
          1. model_module.summary()
          2. model_module.model.fit(...) with EarlyStopping
          3. Evaluate on test set
          4. Save model to disk (as <model_name>.h5)
          5. Save performance plots
          6. Save final‐metrics.txt

        Returns a dict of metrics & filepaths.
        """
        # 1) Show architecture
        print(f"\n--- Training '{self.model_module.model_name}' ---")
        self.model_module.summary()

        # 2) Get data (may be NumPy arrays or a tf.data.Dataset)
        train_data = self.data_module.get_train_data()
        val_data = self.data_module.get_val_data()
        test_data = self.data_module.get_test_data()

        # 3) EarlyStopping callback
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )

        # 4) Fit
        if isinstance(train_data, tf.data.Dataset):
            # train_data is a tf.data.Dataset of (x, y) pairs (already batched)
            train_ds = train_data
            val_ds = val_data  # also a tf.data.Dataset
            history = self.model_module.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                callbacks=[early_stop],
                verbose=1,
            )
        else:
            # train_data is (X_train, Y_train) as NumPy arrays
            X_train, Y_train = train_data
            X_val, Y_val = val_data
            history = self.model_module.model.fit(
                X_train,
                Y_train,
                validation_data=(X_val, Y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=1,
            )

        # 5) Evaluate
        print(f"\n--- Evaluating '{self.model_module.model_name}' on test set ---")
        if isinstance(test_data, tf.data.Dataset):
            test_ds = test_data
            test_loss, test_acc = self.model_module.model.evaluate(test_ds)
        else:
            X_test, Y_test = test_data
            test_loss, test_acc = self.model_module.model.evaluate(X_test, Y_test)

        print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")

        # 6) Save model (HDF5)
        model_path = self.save_dir / f"{self.model_module.model_name}.h5"
        self.model_module.model.save(model_path)
        print(f"Saved model weights to: {model_path}")

        # 7) Save plots
        plot_path = self._save_plots(history.history)
        print(f"Saved training plots to: {plot_path}")

        # 8) Save final‐metrics.txt
        final_metrics_path = self._save_final_metrics(history.history, test_loss, test_acc)
        print(f"Saved final metrics text file to: {final_metrics_path}")

        return {
            "history": history.history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "model_path": model_path,
            "plot_path": plot_path,
            "final_metrics_path": final_metrics_path,
        }

    def _save_plots(self, history_dict: dict):
        """
        Given history.history, create a single figure with two subplots (accuracy & loss)
        and save under save_dir/<model_name>_performance.png.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        ax1.plot(history_dict["accuracy"], label="train_acc")
        ax1.plot(history_dict["val_accuracy"], label="val_acc")
        ax1.set_title(f"{self.model_module.model_name}: Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Loss
        ax2.plot(history_dict["loss"], label="train_loss")
        ax2.plot(history_dict["val_loss"], label="val_loss")
        ax2.set_title(f"{self.model_module.model_name}: Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        filename = self.save_dir / f"{self.model_module.model_name}_performance.png"
        plt.savefig(filename)
        plt.close(fig)
        return filename

    def _save_final_metrics(self, history_dict: dict, test_loss: float, test_acc: float):
        """
        Writes out a text file under save_dir/<model_name>_final_metrics.txt containing:
          - Final training loss & accuracy (last epoch of history_dict)
          - Final validation loss & accuracy (last epoch)
          - Test loss & test accuracy
        """
        filepath = self.save_dir / f"{self.model_module.model_name}_final_metrics.txt"

        # Extract the last epoch’s metrics
        num_epochs = len(history_dict["loss"])
        final_train_loss = history_dict["loss"][num_epochs - 1]
        final_train_acc = history_dict["accuracy"][num_epochs - 1]
        final_val_loss = history_dict["val_loss"][num_epochs - 1]
        final_val_acc = history_dict["val_accuracy"][num_epochs - 1]

        # Write to a simple text file
        with open(filepath, "w") as f:
            f.write(f"Model: {self.model_module.model_name}\n")
            f.write("Final Training Loss:    {:.4f}\n".format(final_train_loss))
            f.write("Final Training Accuracy:{:.4f}\n".format(final_train_acc))
            f.write("Final Validation Loss:  {:.4f}\n".format(final_val_loss))
            f.write("Final Validation Accuracy:{:.4f}\n".format(final_val_acc))
            f.write("\n")
            f.write("Test Loss:             {:.4f}\n".format(test_loss))
            f.write("Test Accuracy:         {:.4f}\n".format(test_acc))

        return filepath
