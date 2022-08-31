import numpy as np
from tensorflow import keras
from keras.utils import to_categorical


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
