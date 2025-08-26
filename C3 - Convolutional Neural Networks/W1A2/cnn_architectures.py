# cnn_architectures.py

import tensorflow as tf


def sequential_conv_model(input_shape, is_binary=False, units=6):
    if is_binary:
        act_func = "sigmoid"
        units = 1
    else:
        act_func = "softmax"
        # units remains as provided
    model = tf.keras.Sequential(
        [
            tf.keras.layers.ZeroPadding2D(padding=3, input_shape=input_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1)),
            tf.keras.layers.BatchNormalization(axis=3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units, activation=act_func),
        ]
    )
    return model


def api_conv_model(input_shape, is_binary=False, units=6):
    """
    Same layers as your Sequential model, but built via the Functional API.

    Arguments:
      input_shape: tuple, e.g. (64, 64, 3)
      is_binary: if True, forces units=1 and activation="sigmoid"
                 otherwise activation="softmax"
      units: number of output neurons (ignored if is_binary=True)

    Returns:
      model: a tf.keras.Model with the identical layer stack as your Sequential version.
    """
    if is_binary:
        act_func = "sigmoid"
        units = 1
    else:
        act_func = "softmax"
    inputs = tf.keras.Input(shape=input_shape)

    # First conv‐pool
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), padding="same")(inputs)
    A1 = tf.keras.layers.ReLU()(Z1)
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding="same")(A1)

    # Second conv‐pool
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding="same")(P1)
    A2 = tf.keras.layers.ReLU()(Z2)
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding="same")(A2)

    # Flatten + Dense
    F = tf.keras.layers.Flatten()(P2)
    outputs = tf.keras.layers.Dense(units=units, activation=act_func)(F)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
