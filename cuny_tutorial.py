#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


redwine_url = "https://raw.githubusercontent.com/stanleycai123/ctp_datascience_instructor/master/winequality-red.csv"
redwine_data = pd.read_csv(redwine_url, sep=";")
whitewine_url = "https://raw.githubusercontent.com/stanleycai123/ctp_datascience_instructor/master/winequality-white.csv"
whitewine_data = pd.read_csv(whitewine_url, sep=";")
frames = [redwine_data, whitewine_data]
wine_data = pd.concat(frames, axis=0)

redwine_data.shape
whitewine_data.shape

# Question 1: How to check dimensions of merged dataframe (wine_data)?
wine_data.shape

wine_data.head(5)
wine_data.dtypes
wine_data.hist()

wine_data.columns

X = wine_data.loc[:, wine_data.columns != 'alcohol']
y = wine_data.loc[:, 'alcohol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = LinearRegression().fit(X_train, y_train)

reg.score(X_train, y_train)
reg.coef_
wine_data.columns

# Question 2: If fixed acidity increases by 1, how much does the predicted alcohol content increase?

y_pred = reg.predict(X_test)
test_error = mean_squared_error(y_test, y_pred)
test_error

y_train_pred = reg.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)
train_error

# Question 3: Is there overfitting?