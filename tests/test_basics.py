import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.utils import to_categorical
from numpy.typing import NDArray
from typing import Any


def test_basics():
    assert True, "The tests pass"


def test_numpy_1d():
    a = np.array([1, 2, 3, 4])
    assert list(a) == [1, 2, 3, 4]
    assert list(a*2) == [2, 4, 6, 8]


def test_numpy_2d():
    a = np.array([[1, 2], [3, 4]])
    assert np.array_equal(a, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(a*2, np.array([[2, 4], [6, 8]]))


def test_numpy_cast():
    a = np.array([[32, 128], [16, 256]])
    a_converted = a.astype(np.float32) / 256.0
    a_expected = np.array([[0.125, 0.5], [0.0625, 1.0]])
    assert np.array_equal(a_converted, a_expected)


def test_one_hot_encoding():
    """Converts an array of values into one hot encoding.
    https://www.youtube.com/watch?v=v_4KWmkwmsU
    """
    actual = to_categorical([0, 1, 2], 3)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.array_equal(actual, expected)


def test_numpy_types():
    any_array = np.array([0, 1, 2, 3])  # This is an NDArray[Any]
    _any_array_1: NDArray[Any] = any_array  # The actual type
    _any_array_2: NDArray[np.uint] = any_array  # Demonstration that this is an Any.
    _any_array_3: NDArray[np.float64] = any_array  # Demonstration that this is an Any.
    assert any_array.dtype == np.dtype(
        'int64'), "The actual type is an int64, even though it's marked as an Any"

    uint_array = np.array([0, 1, 2, 3], np.uint)
    assert uint_array.dtype == np.dtype('uint')
    _uint_arr_actual_type: NDArray[np.uint] = uint_array
    _uint_arr_expect_err: NDArray[np.int32] = uint_array  # type: ignore

    nested_array = np.array([[0, 1], [2, 3], [4, 5]], np.uint)
    assert nested_array.dtype == np.dtype('uint')
    assert nested_array.shape == (3, 2)


def test_layer_gap():
    x = tf.constant([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9], [10, 11]],
    ])
    y = tf.keras.layers.GlobalAveragePooling1D()(x)
    assert np.array_equal(y.numpy(), [
        [1,  2],  # Average of [0, 1], [2, 3] = (0 + 2) / 2 and (1 + 3) / 2
        [5,  6],
        [9, 10]]
    )


def test_layer_embedding():
    tf.random.set_seed(0)
    embed = keras.layers.Embedding(4, 5)
    embedded_results = embed(tf.constant([[1, 2, 3], [2, 3, 1]]))
    gap_results = tf.keras.layers.GlobalAveragePooling1D()(embedded_results)

    print("embed.embeddings", embed.embeddings)
    print("embedded_results", embedded_results)
    print("gap_results", gap_results)
    assert np.array_equal(
        gap_results.numpy()[0],
        gap_results.numpy()[1]
    ), """
        The average of the results should be the same, since the embedding vectors
        contain the same values just out of order.
    """
