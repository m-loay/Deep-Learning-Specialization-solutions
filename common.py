from pathlib import Path
import os
import numpy as np


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


def multiple_test(test_cases, target):
    print("Running multiple_test...\n")

    for case in test_cases:
        name = case["name"]
        input_args = case["input"]
        expected = case["expected"]
        error_msg = case["error"]
        success = True

        print(f"Running test: {name}...")

        try:
            output = target(*input_args)

            # === Case 1: output is a list of (X, Y) tuples ===
            if isinstance(expected, list) and all(isinstance(pair, tuple) for pair in expected):
                for i, ((expected_X, expected_Y), (out_X, out_Y)) in enumerate(
                    zip(expected, output)
                ):
                    if not isinstance(out_X, np.ndarray) or not isinstance(out_Y, np.ndarray):
                        print(f"[FAIL] Batch {i}: {error_msg} (Not numpy arrays)")
                        success = False
                        continue

                    if out_X.shape != expected_X.shape or out_Y.shape != expected_Y.shape:
                        print(f"[FAIL] Batch {i}: {error_msg} (Wrong shape)")
                        success = False
                        continue

                    if not np.allclose(out_X, expected_X):
                        print(f"[FAIL] Batch {i} X: {error_msg} (Wrong values)")
                        success = False
                        continue

                    if not np.array_equal(out_Y, expected_Y):
                        print(f"[FAIL] Batch {i} Y: {error_msg} (Wrong values)")
                        success = False
                        continue

                if success:
                    print(f"All Batches passed ")
                else:
                    print(f"Failed Batches")

            # === Case 2: output is a dictionary of arrays ===
            elif isinstance(expected, dict):
                for key in expected:
                    if key not in output:
                        print(f"[FAIL] Missing key '{key}' in output")
                        success = False
                        continue

                    expected_val = expected[key]
                    out_val = output[key]

                    if not isinstance(out_val, np.ndarray):
                        print(f"[FAIL] {key}: {error_msg} (Not a numpy array)")
                        success = False
                        continue

                    if out_val.shape != expected_val.shape:
                        print(
                            f"[FAIL] {key}: {error_msg} (Wrong shape {out_val.shape} != {expected_val.shape})"
                        )
                        success = False
                        continue

                    if not np.allclose(out_val, expected_val):
                        print(f"[FAIL] {key}: {error_msg} (Wrong values)")
                        success = False
                        continue

                if success:
                    print(f"All Batches passed ")
                else:
                    print(f"Failed Batches")

            # === Fallback: Simple array or scalar comparison ===
            else:
                if isinstance(expected, np.ndarray):
                    if not np.allclose(output, expected):
                        print(f"[FAIL]: {error_msg} (Wrong array output)")
                    else:
                        print("[PASS]")
                elif output != expected:
                    print(f"[FAIL]: {error_msg} (Mismatched values)")
                else:
                    print("[PASS]")

        except Exception as e:
            print(f"[ERROR] {name}: Exception occurred - {e}")
