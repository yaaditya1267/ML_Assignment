# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:08:55 2020

@author: Anmol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):  # for train
        self.w_ = np.zeros(1 + x.shape[1])  #
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = float(self.eta * (target - self.predict(xi)))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)

        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


df = pd.read_csv('D:\Edge Download\Iris\iris.csv', header=None)
# print(df)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values
print(x)
plt.scatter(x[:50, 0], x[:50, 1], color='blue', marker='o', label='Setosa')  # 50 values in rows
plt.scatter(x[50:100, 0], x[50:100, 1], color='red', marker='x', label='Versicolor')
plt.xlabel('sepal-length')
plt.ylabel('petal-length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(0.1,5)
ppn.fit(x, y)
print(ppn.w_)

plt.plot(range(1, len(ppn.errors) + 1), ppn.errors)

plt.xlabel('Attempts')
plt.ylabel('Number Of Classification')
plt.show()

if ppn.predict([5.0, 1.4]) == -1:
    print("Setosa")
else:
    print("Not Setosa")
