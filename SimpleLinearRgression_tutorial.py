# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:19:22 2020

@author: Akhil
"""

# SIngle linear regression formulae --> y= mx + c

dataset = r"E:\udemy\000000 Datasets\P14-Part2-Regression\P14-Part2-Regression\Section 6 - Simple Linear Regression\Python\Salary_Data.csv"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(dataset)
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regression  = LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)

plt.scatter(x_train,y_train,color="r")
plt.plot(x_train,regression.predict(x_train))
plt.plot(x_test,y_pred)
plt.show()

#if u want to predict seperatley with number use below code:
regression.predict([[5.5]])