# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:55:57 2017

@author: DKIM
"""
# load libraries
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#GridSearchCV template was taken from the Boston Housing project from the Udacity Machine learning Nanodegree

# this is a parameter tuning of a logisitic regression model using gridsearch CV
def fit_model_log_reg(X, y):
    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # initialize the model
    log_reg = LogisticRegression()

    # initalize the parameter and ranges to be used for parameter tuning
    params = {'penalty': ['l1','l2']
                , 'random_state':[319]}
               # , 'solver': ['liblinear','sag']
               # , 'max_iter': range(100,200,50)}

    # tune and fit the model
    grid = GridSearchCV(estimator = log_reg, param_grid = params, cv = cv_sets)  
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_

def fit_model_d_trees(X, y):

    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # initialize the model
    d_tree = DecisionTreeClassifier()

    # initalize the parameter and ranges to be used for parameter tuning
    params = {'criterion':['gini','entropy']
                    ,'max_depth': range(1,10)
                    #,'min_samples_split': range(1,5)
                    ,'min_samples_leaf': range(1,5)
                    ,'max_features': ['auto']
                    , 'random_state':[319]}

    # tune and fit the model
    grid = GridSearchCV(estimator = d_tree, param_grid = params, cv = cv_sets)
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_

def fit_model_rf_trees(X, y):
    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # initialize the model
    rf_tree = RandomForestClassifier()

    # initalize the parameter and ranges to be used for parameter tuning
    params = {'n_estimators': [50]#range(10,101,10)
                    ,'max_depth': range(1,5)
                    ,'bootstrap':[True,False]
                    #,'oob_score':[True,False]
                    #,'min_samples_split': [1]#range(1,6)
                    ,'min_samples_leaf': range(1,6)#range(1,6)
                    ,'max_features': ['auto','sqrt','log2']
                    , 'random_state':[319]}

    # tune and fit the model                    
    grid = GridSearchCV(estimator = rf_tree, param_grid = params, cv = cv_sets)
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_

def fit_model_adaboost(X, y):
    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # initialize the model
    adaboost = AdaBoostClassifier()

    # initalize the parameter and ranges to be used for parameter tuning
    params = {'n_estimators': [10]#range(5,51,5)
                    ,'learning_rate': [0.001,0.01,0.1,1]#np.linspace(0.05,1.0,20)
                    , 'random_state':[319]}

    # tune and fit the model
    grid = GridSearchCV(estimator = adaboost, param_grid = params, cv = cv_sets)
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_

def fit_model_xgboost(X, y):
    # split the dataset for cross validation
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # initialize the model
    xgboost = xgb.XGBClassifier(silent=False)

    # tune and fit the model
    params = {'n_estimators': [50,100]#[10,100,10]#range(1,10)
                    ,'learning_rate': [0.0005,0.0001,0.00001]
                    , 'max_depth': [6]#range(1,8)
                    , 'min_child_weight': [1]
                    #, 'max_delta_step': range(1,10)
                    , 'n_jobs': [4]
                    #, 'gamma': range(0,3)
                    , 'random_state':[319]}

    # tune and fit the model
    grid = GridSearchCV(estimator = xgboost, param_grid = params, cv = cv_sets)
    grid = grid.fit(X, y)

    # Return the best model and pass the parameters used to fit that model
    return grid.best_estimator_, grid.best_params_

# The purpose of this function is test the second approach of retraining using the latest data to predict the next week of games
# this function will retrain with the recent data up to the prior week and predict the week of games of the gameweek variable 
def weeklyIteration(gameweek, data):
    seednumber = 319    
    
    # split the data into training and testing dataset
    # the training dataset will include data prior to the gameweek variable
    data_train = data[data['gameweek'] < gameweek]
    data_train.reset_index()
    data_test = data[data['gameweek'] == gameweek]
    data_test.reset_index()
    
    # split the training and testing dataset into features and target
    features_train = data_train.drop('spreadflag', axis = 1)
    target_train = data_train['spreadflag']
    
    features_test = data_test.drop('spreadflag', axis = 1)
    target_test = data_test['spreadflag']

    # load the xgboost library
    import os
    mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
    os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    import xgboost as xgb
    
    # initalize and fit the model
    xgb_model = xgb.XGBClassifier(random_state = seednumber)
    xgboost = xgb_model.fit(features_train, target_train)

    # convert the data into DMatrix, so it can work with xgb.cv
    dtrain = xgb.DMatrix(features_train, label=target_train)
    dtest = xgb.DMatrix(features_test, label=target_test)
    
    # set the parameters for use in the xgboost cross validation
    params = {'learning_rate': 0.0005, 'min_child_weight': 1,
                 'objective': 'binary:logistic', 'max_depth':6,'subsample':1.0,'nthread':4,'random_state':319,'silent':True}
    
    # process xgboost cross validation
    cv_xgb = xgb.cv(params = params, dtrain = dtrain, num_boost_round = 100, nfold = 10,
                    metrics = ['auc','error'],early_stopping_rounds = 50)
    
    # create the optimmized xgboost model
    final_xgboost = xgb.train(params, dtrain, num_boost_round = 50)
    
    # generate the predictions made for the training and testing datasets
    from sklearn.metrics import accuracy_score
    predictions_train = final_xgboost.predict(dtrain)
    predictions_train = [round(value) for value in predictions_train]
    
    predictions_test = final_xgboost.predict(dtest)
    predictions_test =  [round(value) for value in predictions_test]
    
    # calculate the training and testing accuracy
    xgboost_tuned2_train_accuracy = accuracy_score(target_train, predictions_train)
    xgboost_tuned2_test_accuracy = accuracy_score(target_test, predictions_test)
    
    # return the accuracy and predictions made for the training and testing datasets
    return xgboost_tuned2_train_accuracy, xgboost_tuned2_test_accuracy, predictions_test, target_test

     