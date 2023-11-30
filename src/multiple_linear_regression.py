import numpy as np

class MultipleLinearRegression:
    def __init__ (self, default_intercept = 0, default_coefficients = None):
        self.intercept = default_intercept
        self.coefficients = default_coefficients

    def train(self, x, y): 
        xt = np.matrix.transpose(x)
        w = np.matmul((np.matmul(np.linalg.inv(np.matmul(xt, x)), xt), y)
        return w

    def predict(self, x, w):
        y = np.matmul(x, w)
        return y

    def 
