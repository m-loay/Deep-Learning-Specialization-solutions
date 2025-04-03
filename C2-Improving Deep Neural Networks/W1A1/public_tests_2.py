import numpy as np


def initialize_parameters_zeros_test(target):
    print("Running initialize_parameters_zeros_test...")
    layer_dims = [3, 2, 1]
    expected_output = {
        "W1": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "b1": np.array([[0.0], [0.0]]),
        "W2": np.array([[0.0, 0.0]]),
        "b2": np.array([[0.0]]),
    }

    output = target(layer_dims)

    for key in expected_output:
        if output[key].dtype != expected_output[key].dtype:
            print(f"[FAIL] {key}: Datatype mismatch")
        elif output[key].shape != expected_output[key].shape:
            print(f"[FAIL] {key}: Shape mismatch")
        elif not np.array_equal(output[key], expected_output[key]):
            print(f"[FAIL] {key}: Wrong output")
        else:
            print(f"[PASS] {key}")


def initialize_parameters_random_test(target):
    print("Running initialize_parameters_random_test...")
    layer_dims = [3, 2, 1]
    np.random.seed(3)
    expected_output = {
        "W1": np.array(
            [[17.88628473, 4.36509851, 0.96497468], [-18.63492703, -2.77388203, -3.54758979]]
        ),
        "b1": np.array([[0.0], [0.0]]),
        "W2": np.array([[-0.82741481, -6.27000677]]),
        "b2": np.array([[0.0]]),
    }

    np.random.seed(3)
    output = target(layer_dims)

    for key in expected_output:
        if output[key].dtype != expected_output[key].dtype:
            print(f"[FAIL] {key}: Datatype mismatch")
        elif output[key].shape != expected_output[key].shape:
            print(f"[FAIL] {key}: Shape mismatch")
        elif not np.allclose(output[key], expected_output[key], atol=1e-7):
            print(f"[FAIL] {key}: Wrong output")
        else:
            print(f"[PASS] {key}")


def initialize_parameters_he_test(target):
    print("Running initialize_parameters_he_test...")
    layer_dims = [3, 1, 2]
    np.random.seed(2)
    expected_output = {
        "W1": np.array([[1.46040903, 0.3564088, 0.07878985]]),
        "b1": np.array([[0.0]]),
        "W2": np.array([[-2.63537665], [-0.39228616]]),
        "b2": np.array([[0.0], [0.0]]),
    }

    np.random.seed(2)
    output = target(layer_dims)

    for key in expected_output:
        if output[key].dtype != expected_output[key].dtype:
            print(f"[FAIL] {key}: Datatype mismatch")
        elif output[key].shape != expected_output[key].shape:
            print(f"[FAIL] {key}: Shape mismatch")
        elif not np.allclose(output[key], expected_output[key], atol=1e-7):
            print(f"[FAIL] {key}: Wrong output")
        else:
            print(f"[PASS] {key}")
