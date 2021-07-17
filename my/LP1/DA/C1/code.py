#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:34:19 2020

@author: srushti
"""
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


dat=pd.read_csv('Iris.csv')


dat[0:10]


dat.shape 
list(dat.columns)

dat.dtypes

dat['x1'].describe() 
dat['x2'].describe() 
dat['x3'].describe() 
dat['x4'].describe() 

dat.mean()


plt.hist(dat['x1'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x2'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x3'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()


plt.hist(dat['x4'],bins=30)           ##############plot histogram
plt.ylabel('No of times')
plt.show()

sns.boxplot(y=dat['x1'])
sns.boxplot(y=dat['x2'])
sns.boxplot(y=dat['x3'])
sns.boxplot(y=dat['x4'])


dat.max()
dat.min()
sns.boxplot(x=dat['class'],y=dat['x2'])

import statistics
statistics.pstdev(dat['x1'])

sns.boxplot(data=dat.ix[:,0:4])  

sns.boxplot(x=dat['class'],y=dat['x1'])

sns.boxplot(x=dat['class'],y=dat['x3'])

sns.boxplot(x=dat['class'],y=dat['x4'])
