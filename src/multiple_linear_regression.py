import numpy as np

class MultipleLinearRegression:
    def __init__ (self, default_intercept = 0, default_coefficients = None):
        self.intercept = default_intercept
        self.coefficients = default_coefficients

    def getWeights()
        xt = np.matrix.transpose(x)
        w = np.matmul((np.matmul(np.linalg.inv(np.matmul(xt, x)), xt), y)
        return w
    
    def train(self, x, y): 
        getWeights()

    def predict(self, x, w):
        y = np.matmul(x, w)
        return y
