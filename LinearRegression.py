import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes


### Loading the diabetes data

data = load_diabetes()

## Splitting the data 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.25,random_state=32)

##  Modelling
from sklearn.linear_model import LinearRegression

linear  = LinearRegression()

linear.fit(X_train, y_train)

## Make Prediction

y_pred = linear.predict(X_test)

plt.plot(y_test,y_pred,'.')

## Plotting the line
x =np.linspace(0,330,100)
y = x
plt.plot(x, y)
plt.show()
