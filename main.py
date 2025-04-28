import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from LinearRegression import LinearRegression as lr
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33,random_state = 42)
plt.scatter(X,y) #Distribution of the data

model = lr()
model.fit(X_train,y_train)
y_head = model.predict(X_test)
compares = pd.concat([pd.DataFrame(y_head.reshape(-1,1),columns = ["Predicted"]),
                      pd.DataFrame(y_test.reshape(-1,1),columns = ["Real"])],axis = 1)

#Bias, coefficents and sample deflection
print("Intercept:",model.intercept_)
print("Coefficients:",model.coeff_)
print("Deflection:",y_test[0] - y_head[0])

#Sklearn Lib
sklearnModel = LinearRegression()
sklearnModel.fit(X_train,y_train)
predict = sklearnModel.predict(X_test)

#Comparing the loss
print("MSE of sklearn model:",np.sum(np.square(y_test-predict))/len(y_head))
print("Mean Squared Error(MSE):",np.sum(np.square(y_test - y_head))/len(y_head))
plt.plot(X_test,y_head,"red")
plt.title("Linear Regression Plot")
plt.show()
