import numpy as np

class MultipleLinearRegression:
    def __init__ (self, default_intercept = 0, default_coefficients = None, w):
        self.intercept = default_intercept
        self.coefficients = default_coefficients

    def getWeights()
        return w
    
    def train(self, x, y): 
        xt = np.matrix.transpose(x)
        w = np.matmul((np.matmul(np.linalg.inv(np.matmul(xt, x)), xt), y)
        self.weights = w
        #change x to add the 1s
        #comment that the data is expected in a specific format

    def predict(self, x, w):
        y = np.matmul(x, w)
        return y
