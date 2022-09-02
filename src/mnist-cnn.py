#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
import tensorflow as tf
import tensorflowjs as tfjs  # type: ignore

from typing import Any, Tuple
from tensorflow import keras
from tensorflow.keras import datasets, utils, layers  # type: ignore

root_dir = (Path(__file__).parent / "..").resolve()
model_dir = root_dir / "models/mnist-cnn"
log_dir = root_dir / "models/mnist-cnn/logs"

# trainX shape: (60000, 28, 28)
# trainY shape: (60000,)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train_flat = y_train
y_test_flat = y_test
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


def output_image(images: NDArray[np.float32], labels: NDArray[np.uint], index: int) -> None:
    image = images[index]
    label = labels[index]
    height = x_train.shape[1]
    width = x_train.shape[2]

    string = ""

    for i in range(height):
        for j in range(width):
            if image[i][j] > 0.5:
                string += "X"
            else:
                string += "."
        string += "\n"

    print(string)
    print("\n^ This image is labeled \"{}\"\n".format(label))

    string


RawData = Tuple[NDArray[np.uint], NDArray[np.uint]]


def build_model() -> Any:
    if model_dir.exists():
        print("Model found at {}".format(model_dir))
        model = keras.models.load_model(model_dir)
        print("To re-build, run")
        print("  rm -rf {}".format(model_dir))
        return model

    print("No model found, creating a new one")
    # Create the feed forward network.
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    print("Compiling the model")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Log dir")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    print("Training.")

    model_result = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=128,
        epochs=5,
        callbacks=[tensorboard_callback]
    )
    # print(model_result.history)

    print("Saving the model to: {}".format(model_dir))
    model.save(model_dir)
    print("Outputing tensorflowjs model as well")

    tfjs.converters.save_keras_model(model, model_dir)

    return model


model = build_model()
model.summary()


def make_predictions(model: Any) -> None:
    count = 10
    predictions = model.predict(x_test[:count])

    for i in range(count):
        prediction = predictions[i]
        output_image(x_test, y_test_flat, i)

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


make_predictions(model)

print("Completed mnist script")
