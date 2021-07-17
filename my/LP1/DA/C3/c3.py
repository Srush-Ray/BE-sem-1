#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:05:17 2020

@author: srushti
"""

import math
import pandas as pd
import numpy as np


from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
#Generate polynomial and interaction features.

train = pd.read_csv("./Train.csv")
test = pd.read_csv("./Test.csv")



print(train.head())
print(test.head())

print(train.info())
print(test.info())


print(train['Item_Fat_Content'].unique())

train['Item_Fat_Content'].replace(to_replace='low fat', value='Low Fat', inplace=True )
train['Item_Fat_Content'].replace(to_replace='LF', value='Low Fat', inplace=True )
train['Item_Fat_Content'].replace(to_replace='reg', value='Regular', inplace=True )
test['Item_Fat_Content'].replace(to_replace='low fat', value='Low Fat', inplace=True )
test['Item_Fat_Content'].replace(to_replace='LF', value='Low Fat', inplace=True )
test['Item_Fat_Content'].replace(to_replace='reg', value='Regular', inplace=True )

col_enc = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Location_Type', 'Outlet_Type']
for x in col_enc:
    train[x], _ = pd.factorize(train[x])
    print(pd.factorize(train[x]))
    test[x], _ = pd.factorize(test[x])  
    
    
test.isnull().sum()

from sklearn.linear_model import LinearRegression
train_sub = train.drop(['Outlet_Size'], axis = 1)
print( train_sub[train_sub["Item_Weight"].isnull()])
train_sub_test = train_sub[train_sub["Item_Weight"].isnull()]
train_sub = train_sub.dropna()
y_train = train_sub["Item_Weight"]
X_train = train_sub.drop("Item_Weight", axis=1)
X_test = train_sub_test.drop("Item_Weight", axis=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
train.loc[train.Item_Weight.isnull(), 'Item_Weight'] = y_pred


test_sub = test.drop(['Outlet_Size'], axis = 1)
test_sub_test = test_sub[test_sub["Item_Weight"].isnull()]
test_sub = test_sub.dropna()
y_test = test_sub["Item_Weight"]
X_test = test_sub.drop("Item_Weight", axis=1)
X_test_test = test_sub_test.drop("Item_Weight", axis=1)
lr = LinearRegression()
lr.fit(X_test, y_test)
y_pred = lr.predict(X_test_test)
test.loc[test.Item_Weight.isnull(), 'Item_Weight'] = y_pred

train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True )
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True )
train['Outlet_Size'], _ = pd.factorize(train['Outlet_Size'])
test['Outlet_Size'], _ = pd.factorize(test['Outlet_Size'])  

from sklearn.model_selection import train_test_split
X = train.drop(['Item_Outlet_Sales'], axis = 1)
y = train['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

reg = GradientBoostingRegressor(random_state = 42)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))




from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

rf = RandomForestRegressor(max_depth = 2, random_state = 42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))



# Decision Tree
dt = DecisionTreeRegressor(random_state = 42)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))



# K Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors = 2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))



rng = np.random.RandomState(42)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Root mean squared error: ', math.sqrt(mean_squared_error(y_test, predictions)))
print('Mean absolute error: ', mean_absolute_error(y_test, predictions))
print('Coefficient of determination (R2): ', r2_score(y_test, predictions))