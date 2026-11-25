# Multiple linear regression with an intercept.
import numpy as np

class MLR:
    def __init__(self):
        self.coef = None

    def fit(self, x_set, y_set):
        x_set = np.asarray(x_set, dtype=float)
        y_set = np.asarray(y_set, dtype=float).reshape(-1)
        if x_set.ndim != 2 or x_set.shape[0] != y_set.shape[0]:
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if x_set.shape[0] == 0:
            raise ValueError("Cannot fit the model with no training rows.")

        x_with_intercept = np.column_stack((np.ones(x_set.shape[0]), x_set))
        self.coef = np.linalg.lstsq(x_with_intercept, y_set, rcond=None)[0]
        return "MLR"

    def predict(self, x_set):
        if self.coef is None:
            raise ValueError("Fit the model before requesting predictions.")

        x_set = np.asarray(x_set, dtype=float)
        if x_set.ndim == 1:
            x_set = x_set.reshape(1, -1)
        if x_set.ndim != 2 or x_set.shape[1] != self.coef.shape[0] - 1:
            raise ValueError("Prediction features do not match the fitted model.")

        x_with_intercept = np.column_stack((np.ones(x_set.shape[0]), x_set))
        return x_with_intercept @ self.coef
