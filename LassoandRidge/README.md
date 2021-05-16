# Lasso and Ridge Regression

- # Ridge Regression
    - Ridge regression is a regularized version of linear regression. This forces the training algorithm not only to fit the data but also to keep the model weights as small as possible.Note that the accrual term should only be added to the cost function during training. After you train the model, you want to use the unregulated performance measure to evaluate the performance of the model.


- # Lasso Regression
    - Least absolute shrinkage and selection operator regression (usually just called lasso regression) is another regularized version of linear regression: just like peak regression, it adds a regularization term to the cost function. , but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm.

## Dataset images

- TV ads vs Sales
![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/6436fa6968edcdfe9212302884346f7bf1ad54ca/NaiveBayes/GNB_MNB_out.PNG)

- Radio ads vs Sales
![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/6436fa6968edcdfe9212302884346f7bf1ad54ca/NaiveBayes/GNB_MNB_out.PNG)

- Newspaper ads vs Sales

![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/6436fa6968edcdfe9212302884346f7bf1ad54ca/NaiveBayes/GNB_MNB_out.PNG)


```
Linear Regression mean_MSE:  -3.072946597100209
```

```
Ridge Regression mean_MSE:  -3.072671338341143
```

```
LASSO Regression mean_MSE:  -3.0414058967513684
```

## As we can see the Lasso Regression performs better than the Linear Model also ensuring that there is no case of overfitting