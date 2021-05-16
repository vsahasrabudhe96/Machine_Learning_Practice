import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOADING THE DATASET

url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv'
df = pd.read_csv(url)

df.drop('Unnamed: 0',axis=1,inplace=True)


print(df.head())

# Let us plot the graph of dependent to independent variable

def scatter_plot(feature, target):
    plt.figure(figsize=(16, 18))
    plt.scatter(df[feature],
                df[target],
                c='orange'
                )
    plt.xlabel("Money Spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()

scatter_plot("TV", "Sales")
scatter_plot("Radio", "Sales")
scatter_plot("Newspaper", "Sales")

# Let us  run the initial Linear Regression model to check the issues faced in that
'''
LINEAR REGRESSION
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X = df.drop('Sales',axis=1)
y = df['Sales'].values.reshape(-1,1)

linreg = LinearRegression()

MSE = cross_val_score(linreg,X,y,scoring="neg_mean_squared_error",cv=5)

mean_MSE = np.mean(MSE)

print('Linear Regression mean_MSE: ',mean_MSE)


'''
RIDGE REGRESSION
'''

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

param_dic_ridge = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_Regression = GridSearchCV(ridge,param_dic_ridge,scoring="neg_mean_squared_error",cv=5)

ridge_Regression.fit(X,y)

print(ridge_Regression.best_params_)
print("Ridge Regression mean_MSE: ",ridge_Regression.best_score_)


'''
LASSO REGRESSION
'''

from sklearn.linear_model import Lasso
lasso = Lasso()

parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regression = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regression.fit(X, y)

print(lasso_regression.best_params_)
print("LASSO Regression mean_MSE: ",lasso_regression.best_score_)