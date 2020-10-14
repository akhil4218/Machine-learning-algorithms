# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:53:15 2020

@author: Akhil
"""

# Multiple linear regression formulae --> y= c + m0x0 + m1x1 + m2x2....

dataset = r"E:\udemy\000000 Datasets\P14-Part2-Regression\P14-Part2-Regression\Section 7 - Multiple Linear Regression\Python\50_Startups.csv"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(dataset)
X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
lc = LabelEncoder()
X[:,3] = lc.fit_transform(X[:,3])
hot = OneHotEncoder(categorical_features=[3])
X = hot.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regression  = LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)
 

