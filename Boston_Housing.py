#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:06:23 2020

@author: jeffbarrecchia
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as ttSplit
import matplotlib.pyplot as plt
import seaborn as sb

boston = load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['Price'] = boston.target

df_x = df.drop(columns = ['Price'])
df_y = df['Price']

x_train, x_test, y_train, y_test = ttSplit(df_x, df_y, train_size = 0.8, random_state = 3)

y_test = list(y_test)
regr = LinearRegression()

regr.fit(x_train, y_train)
predict = regr.predict(x_test)


r_2 = np.mean((predict - y_test)**2)
# sb.pairplot(df)
# corr = df.corr()
# plt.figure(figsize = (10, 10))
# sb.heatmap(corr, annot = True)
# sb.swarmplot('AGE', 'Price', data = df)