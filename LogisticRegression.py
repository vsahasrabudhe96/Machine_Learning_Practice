import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

### Getting the dataset ready

from sklearn.datasets import load_digits

digits = load_digits()

print("Image Data Shape",digits.data.shape)

print("Target/Label Size",digits.target.shape)

### Plotting the labels

plt.figure(figsize=(20,4))
for index, (image,target) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.gray)
    plt.title('Training: %in' % target, fontsize = 20)


### Split the data 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)

###Modelling 

LR = LogisticRegression()
LR.fit(X_train, y_train)
LR.predict(X_test[0].reshape(1,-1))
LR.predict(X_test[0:10])
predictions = LR.predict(X_test)


## Accuracy Score
score = LR.score(X_test, y_test)
print(score)


