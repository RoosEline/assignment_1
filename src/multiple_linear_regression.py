import numpy as np

class MultipleLinearRegression:
    def __init__ (self, default_intercept = 0, default_coefficients = None):
        self.intercept = default_intercept
        self.coefficients = default_coefficients

    def train(self, X, y): 
        #X is 2D numpy array (rows are samples, columns are features)
        #y is 1D numpy array (target values)
        ones_column = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate((ones_column,X), axis=1)

        self.coefficients = np.linalg.inv(X_with_intercept)
    
    
    def predict(self, x):
        '''
        x is a numpy array
        '''
        return self.slope * x + self.intercept
    

    #make weights private --> __weights
    #def get_weights():
    #return set__weights

    #def set_weights(value) should not be in the code cause we are not changing the weights
    #set __weights = value 

    #don't access variables directly because that constrains you 
    #getters and setters, if accessing something outside but not change value: getter, if you do wanna change the value then set (passwords for example only need setters but not getters cause otherwise you would expose the)

    np.transpose()


    import numpy as np

class MultipleLinearRegression:
    def __init__(self, default_intercept=0, default_coefficients=None):
        self.intercept = default_intercept
        self.coefficients = default_coefficients

    def train(self, X, y):
        '''
        X is a 2D numpy array where rows represent samples and columns represent features.
        y is a 1D numpy array containing the target values.
        '''
        ones_column = np.ones((X.shape[0], 1))  # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((ones_column, X), axis=1)  # Add intercept to features

        # Calculate coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

        # Extract intercept and coefficients
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        '''
        X is a 2D numpy array where rows represent samples and columns represent features.
        '''
        ones_column = np.ones((X.shape[0], 1))  # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((ones_column, X), axis=1)  # Add intercept to features

        # Calculate predictions using the coefficients
        return X_with_intercept.dot(np.hstack(([self.intercept], self.coefficients)))
