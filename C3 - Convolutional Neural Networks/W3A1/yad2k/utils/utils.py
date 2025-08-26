"""Draw predicted or ground truth boxes on input image."""

import imghdr
import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K
import numpy as np
from PIL import ImageDraw, ImageFont
from functools import reduce


def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.0
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def scale_boxes(boxes, image_shape):
    """Scales the predicted boxes in order to be drawable on the image"""
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (
        hasattr(get_colors_for_classes, "colors")
        and len(get_colors_for_classes.colors) == num_classes
    ):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image (PIL.Image) using (y_min, x_min, y_max, x_max) format."""

    try:
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf', size=int(np.floor(3e-2 * image.size[1] + 0.5))
        )
    except OSError:
        font = ImageFont.load_default()

    thickness = (image.size[0] + image.size[1]) // 300
    colors = get_colors_for_classes(len(class_names))

    for i, c in enumerate(box_classes):
        box_class = class_names[c]
        box = boxes[i]

        if scores is not None:
            score = scores[i].numpy() if hasattr(scores[i], "numpy") else scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)

        # Use textbbox instead of deprecated textsize
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]

        top, left, bottom, right = box
        top = max(0, int(np.floor(top + 0.5)))
        left = max(0, int(np.floor(left + 0.5)))
        bottom = min(image.size[1], int(np.floor(bottom + 0.5)))
        right = min(image.size[0], int(np.floor(right + 0.5)))

        print(label, (left, top), (right, bottom))

        if top - label_height >= 0:
            text_origin = np.array([left, top - label_height])
        else:
            text_origin = np.array([left, top + 1])

        # Draw box
        for j in range(thickness):
            draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[c])

        # Draw label background
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + [label_width, label_height])], fill=colors[c]
        )

        # Draw label text
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)
