# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:55:41 2020

@author: Anmol
"""

# perceptron.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from perceplearning import Perceptron
#from sklearn.linear_model import Perceptron

class Perceptron(object):
   def __init__(self, rate + 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y):
      """Fit training data
      X : Training vectors, X.shape : [#samples, #features]
      y : Target values, y.shape : [#samples]
      """

      # weights
      self.weight = np.zeros(1 + X.shape[1])

      # Number of misclassifications
      self.errors = []  # Number of misclassifications

      for i in range(self.niter):
         err = 0
         for xi, target in zip(X, y):
            delta_w = self.rate * (target - self.predict(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            err += int(delta_w != 0.0)
         self.errors.append(err)
      return self

   def net_input(self, X):
      """Calculate net input"""
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def predict(self, X):
      """Return class label after unit step"""
      return np.where(self.net_input(X) >= 0.0, 1, -1)
  
    
    
df = pd.read_csv('D:\Edge Download\Iris\iris.csv', header=None)
#print(df.head(5))
print(df.iloc[145:150, 0:5])  
y = df.iloc[0:100, 4].values
#print(y)  
y = np.where(y == 'setosa', -1, 1) 
print(y)

X = df.iloc[0:100, [0, 2]].values
print(X)

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

from perceptron import Perceptron
pn = Perceptron(0.1, 10)
pn.fit(X, y)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()



