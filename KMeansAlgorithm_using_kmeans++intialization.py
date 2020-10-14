# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:35:14 2020

@author: Akhil
"""

path = r"E:\udemy\000000 Datasets\P14-Part4-Clustering\Section 25 - K-Means Clustering\Python\Mall_Customers.csv"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(path)
X = dataset.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
    
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.show()

kmeans  = KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
Y = kmeans.fit_predict(X)

plt.scatter(X[Y==0,0],X[Y==0,1],s=100,label="cluster 1",c="red")
plt.scatter(X[Y==1,0],X[Y==1,1],s=100,label="cluster 2",c="blue")
plt.scatter(X[Y==2,0],X[Y==2,1],s=100,label="cluster 3",c="green")
plt.scatter(X[Y==3,0],X[Y==3,1],s=100,label="cluster 4",c="violet")
plt.scatter(X[Y==4,0],X[Y==4,1],s=100,label="cluster 5",c="cyan")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,label="cluster centroid",c="yellow")
plt.title("Cluster grouping")
plt.legend()
plt.show()
