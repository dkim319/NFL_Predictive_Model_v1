# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:21:23 2017

@author: DKIM
"""

import pandas as pd
import numpy as np

seednumber = 319

data = pd.read_csv('Data.csv')

# replace any null values with 0
data.fillna(0)

# use one-hot coding to replace the favorite and underdog categorical variables
fav_team = pd.get_dummies(data['favorite'])
und_team = pd.get_dummies(data['underdog'])

# use a prefix to distinguish the two categorical variables
fav_team = fav_team.add_prefix('fav_')
und_team = und_team.add_prefix('und_')

# remove the original fields
data = data.drop('favorite', axis = 1)
data = data.drop('underdog', axis = 1)

# add the one-hot coded fields
data = pd.concat([data, fav_team], axis = 1)
data = pd.concat([data, und_team], axis = 1)

#print data.head(5)
#print(data.describe())

# split the dataset into training and testing datasets
data_train = data[data['season'] <= 2015]
data_train.reset_index()
data_test = data[data['season'] == 2016]
data_test.reset_index()

# split training and testing datasets into features and target 
features_train = data_train.drop('spreadflag', axis = 1)
target_train = data_train['spreadflag']

features_test = data_test.drop('spreadflag', axis = 1)
target_test = data_test['spreadflag']
