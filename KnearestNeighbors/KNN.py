import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Loading the data

from sklearn.datasets import load_iris
dataset = load_iris()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(dataset['data'] , dataset['target'],test_size=0.25,random_state=32)

iris_dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),marker='o', hist_kwds={'bins': 20}, s=60,alpha=.8)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


knn.fit(X_train, y_train)


prediction = knn.predict(X_test)
print("Prediction:", prediction)
print("Predicted target name:",dataset['target_names'][0])
