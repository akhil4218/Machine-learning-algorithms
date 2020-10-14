# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 20:38:43 2020

@author: Akhil
"""

# step  -1 Import Libraries
# step  -2 Import Dataset
# step  -3 Handling Missing data
# step  -4 Handling categorical data (label encoding, onehotencoding)
# step  -5 Splitting dataset into training and testing sets
# step  -6 Feature scaling


dataset_file = r"E:\udemy\000000 Datasets\P14-Part1-Data-Preprocessing\P14-Part1-Data-Preprocessing\Section 3 - Data Preprocessing in Python\Python\Data.csv"

# step -1 import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step -2 importing the dataset

data = pd.read_csv(dataset_file)
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

# step -3 handling missing data

# from sklearn.preprocessing import Imputer --> this was about to get deprecated so i used SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NAN,strategy='mean')
# imputer = imputer.fit(X[:,1:])
# X[:,1:] = imputer.transform(X[:,1:])

X[:,1:] = imputer.fit_transform(X[:,1:])

# step -4 handling categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_Encoder_x = LabelEncoder()
X[:,0] = label_Encoder_x.fit_transform(X[:,0])
onehotencoder_x = OneHotEncoder(categorical_features=[0])
X = onehotencoder_x.fit_transform(X).toarray()

label_ENcoder_y = LabelEncoder()
Y[:] = label_ENcoder_y.fit_transform(Y[:])

# step -5 splitiing dataset into training set and testing set
# sklearn.cross_validation is already depricated so we used model_selection

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,train_size = 0.8)


# step - 6 feauture Scaling
"""
The reason for doing feature scaling is generally all the machine learning models will follow 
euclidian distance method to find the distance between two points
Euclidian formulae = square root of ((X2-X1)**2 + (Y2-Y1)**2)
So suppose id one column is having data as age which scale  is 15 - 100
and one column is of salry which scale is 100 k - 100k
then euclidian distance of both column will not be in one scale
so in order to make it as one one scale we will do Feature scaling


they are several ways to do feautre scaling

1) Standardisation


standardasation of (x) = x - mean(x) / standard deviation of (x)

2) Normalization

Normalization of (x) = x - min(x) / max(x) - min(x)

"""

from sklearn.preprocessing import StandardScaler

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit(x_test)
