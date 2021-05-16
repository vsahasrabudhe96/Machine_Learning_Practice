# Lasso and Ridge Regression

- # Ridge Regression
    - Ridge regression is a regularized version of linear regression. This forces the training algorithm not only to fit the data but also to keep the model weights as small as possible.Note that the accrual term should only be added to the cost function during training. After you train the model, you want to use the unregulated performance measure to evaluate the performance of the model.


- # Lasso Regression
    - Least absolute shrinkage and selection operator regression (usually just called lasso regression) is another regularized version of linear regression: just like peak regression, it adds a regularization term to the cost function. , but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm.

## Dataset images

- TV ads vs Sales

![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/e95e8c5387f114bb80c05af3ffe02187e936bf20/LassoandRidge/ads_vs_sales.png)

- Radio ads vs Sales

![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/e95e8c5387f114bb80c05af3ffe02187e936bf20/LassoandRidge/LassoandRidge/radioads_vs_sales.png)

- Newspaper ads vs Sales

![alt text](https://github.com/vsahasrabudhe96/Machine_Learning_Practice/blob/e95e8c5387f114bb80c05af3ffe02187e936bf20/LassoandRidge/LassoandRidge/newspaperads_vs_sales.png)


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