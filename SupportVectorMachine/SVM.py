import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


## Loading the dataset
iris = sns.load_dataset('iris')

print(set(iris['species']))

print(iris.head())

## Plotting the data

sns.pairplot(iris,hue='species',palette='Dark2')
plt.show()
setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],cmap="plasma", shade=True, shade_lowest=False)

plt.show()

### Train test split

from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=32)

### Modelling
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train,y_train)
prediction = svc.predict(X_test)

## Accuracy

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,prediction))
print("\n")
print(classification_report(y_test,prediction))
print("\n")

### Performing a Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)

print('Confusion matrix after tuning',confusion_matrix(y_test,grid_pred))
print("\n")
print('Classification report after tuning',classification_report(y_test,grid_pred))
print("\n")

