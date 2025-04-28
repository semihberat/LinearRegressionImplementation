import numpy as np

class LinearRegression():
  def __init__(self):
    self.intercept_ = 0 #Sum of intercepts (bias)
    self.coeff_ = [] #Coefficients if we have 3 column we have 3 coeff

  def fit(self,X,y):
    #Check whether the data type is numpy.ndarray
    if(type(X) != type(np.array([1]))):
      X_train,y_train = X.values,y.values.reshape(-1,1)
    else:
      X_train,y_train = X,y.reshape(-1,1)
    #picking up the sizes (N = Features, D = Samples)
    N,D = X_train.shape
    #Processing all columns with iterations
    for i in range(D):
      column = X_train[:,i].reshape(-1,1)
      sxx = np.sum((np.square(column))) - np.square(np.sum(column))/N #Sum of squared differences of X
      sxy = np.sum(column*y_train) - (np.sum(column))*(np.sum(y_train))/N #Sum of cross products

      intercept = sxy/sxx
      self.intercept_ += sxy/sxx

      coefficients = np.mean(y_train) - intercept*np.sum(X_train)/N
      self.coeff_.append(coefficients)

    self.coeff_ = np.array(self.coeff_)

  def predict(self,X_test):
    #Predicts: The multidimensional formula that you see in the readme.md
    y_head = np.dot(self.coeff_,X_test.T) + self.intercept_
    return y_head.reshape(-1,1)

