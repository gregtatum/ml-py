#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
import tensorflow as tf

from typing import Any
from tensorflow import keras

# feed forward layers
from tensorflow.keras.models import Sequential  # type: ignore
# fully connected layers
from tensorflow.keras.layers import Dense  # type: ignore
# stochastic gradient descent
from tensorflow.keras.optimizers import SGD  # type: ignore


@dataclass
class ImageData:
    labels: NDArray[np.float32]
    images: NDArray[np.float32]
    number_of_rows: int
    number_of_cols: int
    bytes_per_image: int


root_dir = (Path(__file__).parent / "..").resolve()
model_dir = root_dir / "data/mnist-model"
log_dir = root_dir / "data/mnist-model/logs"


def read_in_images(path: str, labels: NDArray[np.float32]) -> ImageData:
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

        print(" > magic_number", magic_number)
        print(" > number_of_images", number_of_images)
        print(" > number_of_rows", number_of_rows)
        print(" > number_of_cols", number_of_cols)
        print(" > bytes_per_image", bytes_per_image)

        data = np.fromfile(
            file,
            dtype=np.dtype(((np.ubyte, bytes_per_image), number_of_images)),
            count=bytes_per_image
        )

        return ImageData(
            images=data[0],
            labels=labels,
            number_of_rows=number_of_rows,
            number_of_cols=number_of_cols,
            bytes_per_image=bytes_per_image,
        )


def read_in_labels(path: str) -> NDArray[np.float32]:
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
        count = read_i32()

        if magic_number != 2049:
            raise Exception(
                "Attempting to load a file that is not an mnist label file.", path)

        data = np.fromfile(
            file,
            dtype=np.ubyte,
            count=count
        )
        results = np.zeros((count, 10), np.float32)
        for n in range(count):
            results[n][data[n]] = 1.0

        return results


def output_image(image_data: ImageData, index: int) -> None:
    image = image_data.images[index]
    label = image_data.labels[index]
    height = image_data.number_of_rows
    width = image_data.number_of_cols

    string = ""

    for i in range(height):
        for j in range(width):
            index = i * height + j
            if image[index] > 0.5:
                string += "X"
            else:
                string += "."
        string += "\n"

    print(string)
    print("\n^ This image is labeled \"{}\"\n".format(list(label).index(1.0)))

    string


def load_in_test_images() -> ImageData:
    print("Load in test images")
    labels = read_in_labels("./data/t10k-labels-idx1-ubyte")
    return read_in_images("./data/t10k-images-idx3-ubyte", labels)


def load_in_training_images() -> ImageData:
    print("Load in training images")
    labels = read_in_labels("./data/train-labels-idx1-ubyte")
    return read_in_images("./data/train-images-idx3-ubyte", labels)


def build_model(train_data: ImageData, test_data: ImageData) -> Any:
    if model_dir.exists():
        print("Model found at {}".format(model_dir))
        model = keras.models.load_model(model_dir)
        print("To re-build, run")
        print("  rm -rf {}".format(model_dir))
        return model

    print("No model found, creating a new one")
    # Create the feed forward network.
    model = Sequential()

    # Create two "hidden" layers
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))

    # Softmax normalizes the output. The output layer can include negative numbers or
    # positive numbers. These values are applied to e^x which converts them into only
    # positive values along a distribution. Finally, each vector component is divided
    # by the sum of the total e^x values for each component. This means the resulting
    # vector components will sum to be 1, and each component will be in the range 0-1.
    #
    # See https://en.wikipedia.org/wiki/Softmax_function
    model.add(Dense(10, activation="softmax"))

    # Define the optimizer.
    gradient_descent = SGD(learning_rate=0.01)

    print("Compiling the model")
    # https://keras.io/api/models/model_training_apis/
    model.compile(
        # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_crossentropy
        loss="categorical_crossentropy",
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        optimizer=gradient_descent,
        #
        metrics=["accuracy"]
    )

    print("Log dir")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    print("Training.")
    model_result = model.fit(
        x=train_data.images,
        y=train_data.labels,
        validation_data=(test_data.images, test_data.labels),
        epochs=100,
        batch_size=128,
        callbacks=[tensorboard_callback]
    )
    # print(model_result.history)

    print("Saving the model to: {}".format(model_dir))
    model.save(model_dir)
    return model


def make_predictions(model: Any, image_data: ImageData) -> None:
    count = 10
    predictions = model.predict(image_data.images[:count])

    for i in range(count):
        prediction = predictions[i]
        output_image(test_data, i)

        minimum = -1.0
        number = 0
        for j in range(10):
            if prediction[j] > minimum:
                minimum = prediction[j]
                number = j
            print(" {}: {}".format(j, "{0:.3f}".format(prediction[j])))

        print()
        print("\nPrediction:", number)
        print()


train_data = load_in_test_images()
test_data = load_in_training_images()
model = build_model(train_data, test_data)
make_predictions(model, test_data)

print("Completed mnist script")
