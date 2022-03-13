# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:16:01 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datapath="D:\AanshFolder\datasets\Advertising.csv"
data = pd.read_csv(datapath)
data.drop(['Unnamed: 0'],axis=1, inplace = True)
print(data.head())

def scatterplot(feature,target):
    plt.figure(figsize=(8,4))
    plt.scatter(
            data[feature],
            data[target],
            c='black'
            )
    plt.xlabel("Money spent on {} on ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()


scatterplot('TV','sales')    
scatterplot('radio','sales')
scatterplot('newspaper','sales')

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = data.drop(['sales'],axis=1)
y=data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg,Xs,y,scoring='neg_mean_squared_error',cv=5)
mean_MSE = np.mean(MSEs)
print("Mean_MSE:",mean_MSE)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]
ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(Xs,y)
print("Ridge Best Params",ridge_regressor.best_params_)
print("Ridge Best Score",ridge_regressor.best_score_)

from sklearn.linear_model import Lasso

lasso = Lasso()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(Xs,y)
print("Lasso Best Params",lasso_regressor.best_params_)
print("Lasso Best Score",lasso_regressor.best_score_)
