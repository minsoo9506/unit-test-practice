import numpy as np
from src.preprocess.standardize import standardize
import pytest

def test_standardize_result():
    X = np.array([[1,1,1],
                  [3,3,3]])
    expected = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    actual = standardize(X)
    message = f'Expected value is {expected} but actual value is {actual}'
    assert expected == pytest.approx(actual), message

def test_standardize_shape():
    X = np.array([[1,1,1],
                  [1,1,1]])
    expected = X.shape
    actual = standardize(X).shape
    message = f'Expected value is {expected} but actual value is {actual}'
    assert expected == actual, message