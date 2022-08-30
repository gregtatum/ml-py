from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Any
# import tensorflow as tf
# from tensorflow import keras

NDf32 = npt.NDArray[np.float32]


@dataclass
class ImageData:
    labels: list[int]
    images: NDf32
    number_of_rows: int
    number_of_cols: int
    bytes_per_image: int


def read_in_images(path: str, labels: list[int]) -> ImageData:
    print("Reading in images from: ", path)
    with open(path, "rb") as file:
        def read_i32() -> int:
            return int.from_bytes(file.read(4), "big")

        magic_number = read_i32()
        number_of_images = read_i32()
        number_of_rows = read_i32()
        number_of_cols = read_i32()
        bytes_per_image = number_of_rows * number_of_cols

        if magic_number != 2051:
            raise Exception("Attempting to load a file that is not an mnist file.", path)

        print("magic_number", magic_number)
        print("number_of_images", number_of_images)
        print("number_of_rows", number_of_rows)
        print("number_of_cols", number_of_cols)
        print("bytes_per_image", bytes_per_image)

        images: NDf32 = np.empty(
            (number_of_images, bytes_per_image), dtype=np.float32)
        for i in range(number_of_images):
            image: NDf32 = images[i]
            for j in range(bytes_per_image):
                image[j] = int.from_bytes(file.read(1), "big") / 255.0
        return ImageData(
            images=images,
            labels=labels,
            number_of_rows=number_of_rows,
            number_of_cols=number_of_cols,
            bytes_per_image=bytes_per_image,
        )


def read_in_labels(path: str) -> list[int]:
    """
    According to: http://yann.lecun.com/exdb/mnist/

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    """

    print("Reading in labels from: ", path)
    with open(path, "rb") as file:
        def read_i32() -> int:
            return int.from_bytes(file.read(4), "big")

        magic_number = read_i32()
        number_of_items = read_i32()

        if magic_number != 2049:
            raise Exception(
                "Attempting to load a file that is not an mnist label file.", path)

        labels = []
        for i in range(number_of_items):
            labels.append(int.from_bytes(file.read(1), "big"))

        return labels


def output_image(image_data: ImageData, index: int) -> None:
    image = image_data.images[index]
    label = image_data.labels[index]
    height = image_data.number_of_rows
    width = image_data.number_of_cols

    string = ""

    for i in range(height):
        for j in range(width):
            index = i * height + j
            if image[index] > 50.0 / 255.0:
                string += "X"
            else:
                string += "."
        string += "\n"

    print(string)
    print("\n^ This image is labeled \"{}\"\n".format(label))

    string


def load_in_test_images() -> ImageData:
    print("Load in test images")
    labels = read_in_labels("./data/t10k-labels-idx1-ubyte")
    return read_in_images("./data/t10k-images-idx3-ubyte", labels)


def load_in_training_images() -> ImageData:
    print("Load in training images")
    labels = read_in_labels("./data/train-labels-idx1-ubyte")
    return read_in_images("./data/train-images-idx3-ubyte", labels)


test_images = load_in_test_images()
for i in range(10):
    output_image(test_images, i)

# training_images = load_in_training_images()

print("Completed mnist script")
