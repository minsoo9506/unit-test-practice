from src.train.LinearRegression import train_LinearRegression
from data.data_generate import generate_data
import numpy as np
import pytest

class TestTrain:

    def test_train_coef(self):
        expected_bias = 1.0
        X, y, expected_coef =generate_data(n_samples=100, n_features=10, bias=expected_bias)
        model = train_LinearRegression(X, y)
        actual_coef = model.coef_
        actual_intercept = model.intercept_
        assert expected_coef == pytest.approx(actual_coef), 'Wrong Coef!'
        assert expected_bias == pytest.approx(actual_intercept), 'Wrong intercept!'

    def test_train_pred(self):
        X, y, w =generate_data(n_samples=100, n_features=2, bias=1.0)
        test_X = np.array([[1.0, 1.0]])
        test_y = w * test_X + 1.0
        model = train_LinearRegression(X, y)
        pred = model.predict(test_X)
        assert test_y == pytest.approx(pred), 'Wrong prediction!'