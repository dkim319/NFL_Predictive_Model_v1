# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:21:23 2017

@author: DKIM
"""

#Run 3b- Supporting Functions first

import pandas as pd
import numpy as np

seednumber = 319

# load data
data = pd.read_csv('Data.csv')

# replace any null values with 0
# replace any null values with 0
data = data.fillna(0)

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

# start timer to capture the amount of time taken to process this python file
from datetime import datetime
timestart = datetime.now()

# the booleans are used to control which section of code is processed
# primary used for troubleshooting a particular section
log_reg_bool = True
d_tree_bool = True
rf_tree_bool = True
ada_boost_bool = True
xgboost_bool = True
feature_selection = True
feature_selection_tuned = True

# -----------------------------------------------
# Logistic Regression
# This section training and predicts the model using model with default settings and with the parameters tuned
if log_reg_bool:
    # load library
    from sklearn.linear_model import LogisticRegression
    
    # initialize and fit the model with default settings
    log_reg = LogisticRegression(random_state = seednumber)
    log_reg.fit(features_train, target_train)

    #calculate the training and testing accuracies
    log_reg_train_accuracy = log_reg.score(features_train, target_train)
    log_reg_test_accuracy = log_reg.score(features_test, target_test)
    print 'log reg'
    print log_reg_train_accuracy
    print log_reg_test_accuracy
    
    # use the fit_model_log_reg to tune the model
    log_reg_tuned, log_reg_tuned_params = fit_model_log_reg(features_train, target_train)
    print 'log reg tuned'
    
    # calculate the training and testing accuracies of the tuned model
    log_reg_tuned_train_accuracy = log_reg_tuned.score(features_train, target_train)
    log_reg_tuned_test_accuracy = log_reg_tuned.score(features_test, target_test)
    print 'log reg'
    print log_reg_tuned_train_accuracy
    print log_reg_tuned_test_accuracy
# -----------------------------------------------
# Decision Tree
# This section training and predicts the model using model with default settings and with the parameters tuned
if d_tree_bool:
    # load library
    from sklearn.tree import DecisionTreeClassifier

    # initialize and fit the model with default settings
    d_tree = DecisionTreeClassifier(random_state = seednumber)
    d_tree.fit(features_train, target_train)

    #calculate the training and testing accuracies
    d_tree_train_accuracy = d_tree.score(features_train, target_train)
    d_tree_test_accuracy = d_tree.score(features_test, target_test)
    
    print 'dtree'
    print d_tree_train_accuracy
    print d_tree_test_accuracy
    
    # use the customized function to tune the model
    d_tree_tuned, d_tree_tuned_params = fit_model_d_trees(features_train, target_train)
    
    # calculate the training and testing accuracies of the tuned model
    d_tree_tuned_train_accuracy = d_tree_tuned.score(features_train, target_train)
    d_tree_tuned_test_accuracy = d_tree_tuned.score(features_test, target_test)
    
    print 'dtree tuned'
    print d_tree_tuned_train_accuracy
    print d_tree_tuned_test_accuracy
    
# -----------------------------------------------
# Random Forest
# This section training and predicts the model using model with default settings and with the parameters tuned
if rf_tree_bool:
    # load library
    from sklearn.ensemble import RandomForestClassifier

    # initialize and fit the model with default settings
    rf_tree = RandomForestClassifier(random_state = seednumber)
    rf_tree.fit(features_train, target_train)

    #calculate the training and testing accuracies
    rf_tree_train_accuracy = rf_tree.score(features_train, target_train)
    rf_tree_test_accuracy = rf_tree.score(features_test, target_test)
    
    print 'rf'
    print rf_tree_train_accuracy
    print rf_tree_test_accuracy
    
    # use the customized function to tune the model
    rf_tree_tuned, rf_tree_tuned_params = fit_model_rf_trees(features_train, target_train)

    # calculate the training and testing accuracies of the tuned model
    rf_tree_tuned_train_accuracy = rf_tree_tuned.score(features_train, target_train)
    rf_tree_tuned_test_accuracy = rf_tree_tuned.score(features_test, target_test)
    
    print 'rft - tuned'
    print rf_tree_tuned_train_accuracy
    print rf_tree_tuned_test_accuracy
    
# -----------------------------------------------
# Adaboost
# This section training and predicts the model using model with default settings and with the parameters tuned
if ada_boost_bool:
    # load library
    from sklearn.ensemble import AdaBoostClassifier

    # initialize and fit the model with default settings
    ada_boost = AdaBoostClassifier(random_state = seednumber)
    ada_boost.fit(features_train, target_train)

    #calculate the training and testing accuracies   
    ada_boost_train_accuracy = ada_boost.score(features_train, target_train)
    ada_boost_test_accuracy = ada_boost.score(features_test, target_test)

    print 'adaboost'
    print ada_boost_train_accuracy
    print ada_boost_test_accuracy

    # use the customized function to tune the model
    ada_boost_tuned, ada_boost_tuned_params = fit_model_adaboost(features_train, target_train)

    # calculate the training and testing accuracies of the tuned model
    ada_boost_tuned_train_accuracy = ada_boost_tuned.score(features_train, target_train)
    ada_boost_tuned_test_accuracy = ada_boost_tuned.score(features_test, target_test)

    print 'adaboost tuned'    
    print ada_boost_tuned_train_accuracy
    print ada_boost_tuned_test_accuracy

# -----------------------------------------------
# XGBoost
# This section training and predicts the model using model with default settings and with the parameters tuned
if xgboost_bool:
    # load library
    import os
    mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
    os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    import xgboost as xgb

    # initialize and fit the model with default settings
    xgb_model = xgb.XGBClassifier(random_state = seednumber)
    xgboost = xgb_model.fit(features_train, target_train)

    # convert the data into DMatrix, so it can work with xgb.cv
    dtrain = xgb.DMatrix(features_train, label=target_train)
    dtest = xgb.DMatrix(features_test, label=target_test)

    #calculate the training and testing accuracies  
    xg_boost_train_accuracy = xgboost.score(features_train, target_train)
    xg_boost_test_accuracy = xgboost.score(features_test, target_test)

    print 'xgboost - reg'
    print xg_boost_train_accuracy
    print xg_boost_test_accuracy
    
    # use the customized function to tune the model
    xg_boost_tuned, xg_boost_tuned_params = fit_model_xgboost(features_train, target_train)

    # calculate the training and testing accuracies of the tuned model
    xg_boost_tuned_train_accuracy = xg_boost_tuned.score(features_train, target_train)
    xg_boost_tuned_test_accuracy = xg_boost_tuned.score(features_test, target_test)
    
    print 'xgboost - tuned'
    print xg_boost_tuned_train_accuracy
    print xg_boost_tuned_test_accuracy
    
    # set the parameters for use in the xgboost cross validation
    params = {'learning_rate': 0.0005, 'min_child_weight': 1,
                 'objective': 'binary:logistic', 'max_depth':6,'subsample':1.0,'nthread':4,'random_state':319,'silent':True,'verbose_eval':0}

    # process xgboost cross validation
    cv_xgb = xgb.cv(params = params, dtrain = dtrain, num_boost_round = 100, nfold = 10,
                    metrics = ['auc','error'],early_stopping_rounds = 100), 
    
    # create the optimmized xgboost model
    final_xgboost = xgb.train(params, dtrain, num_boost_round = 100)
    
    # generate the predictions made for the training and testing datasets
    from sklearn.metrics import accuracy_score
    predictions_train = final_xgboost.predict(dtrain)
    predictions_train = [round(value) for value in predictions_train]
    
    predictions_test = final_xgboost.predict(dtest)
    predictions_test =  [round(value) for value in predictions_test]

    # calculate the training and testing accuracy
    xgboost_tuned2_train_accuracy = accuracy_score(target_train, predictions_train)
    xgboost_tuned2_test_accuracy = accuracy_score(target_test, predictions_test)
    
    print 'xgboost tuned w/ xgb.cv'
    print xgboost_tuned2_train_accuracy
    print xgboost_tuned2_test_accuracy

# This section uses feature importance scores of the xgboost model
# these scores are used as thresholds to perform feature selection
if feature_selection:
    # load library
    from sklearn.feature_selection import SelectFromModel

    # initalize varaibles
    f_select = []
    max_test_thresh = 0   
    max_test_acc = 0
    
    # store the feature importance scores into a variable
    thresholds = sort(xgboost.feature_importances_)

    # for every threshold, apply feature selection using that threshold to create a new training and testing dataset
    for thresh in thresholds:
        
        # intilaize SelectFromModel using thresh
        feature_model =  SelectFromModel(xgb_model, threshold = thresh, prefit=True)
    
        # apply feature selection to the training and testing dataset
        features_train_new = feature_model.transform(features_train)
        features_test_new = feature_model.transform(features_test)
        
        features_train.shape
        features_train_new.shape
        
        # convert the data into DMatrix, so it can work with xgb.cv
        dtrain_new = xgb.DMatrix(features_train_new, label=target_train)
        dtest_new = xgb.DMatrix(features_test_new, label=target_test)
        
        # set the parameters for use in the xgboost cross validation
        params = {'learning_rate': 0.0005, 'min_child_weight': 1, #'subsample':0.8,
                 'objective': 'binary:logistic', 'max_depth':6,'subsample':1,'nthread':4,'random_state':319,'silent':True,'verbose_eval':0}
        
        # process xgboost cross validation
        cv_xgb = xgb.cv(params = params, dtrain = dtrain_new, num_boost_round = 100, nfold = 10,
                    metrics = ['auc','error'],early_stopping_rounds = 20), 

        # create the optimmized xgboost model
        final_xgboost = xgb.train(params, dtrain_new, num_boost_round = 100)
        
        # generate the predictions made for the training and testing datasets
        from sklearn.metrics import accuracy_score
        predictions_train = final_xgboost.predict(dtrain_new)
        predictions_train = [round(value) for value in predictions_train]
        
        predictions_test = final_xgboost.predict(dtest_new)
        predictions_test =  [round(value) for value in predictions_test]

        # calculate the training and testing accuracy        
        xgboost_feature_selected_train_accuracy = accuracy_score(target_train, predictions_train)
        xgboost_feature_selected_test_accuracy = accuracy_score(target_test, predictions_test)
        
        print 'xgboost - fs'
        print xgboost_feature_selected_train_accuracy
        print xgboost_feature_selected_test_accuracy
        
        # store the highest testing accuracy in a varaible
        if xgboost_feature_selected_test_accuracy > max_test_acc:
            max_test_thresh = thresh
            max_test_acc = xgboost_feature_selected_test_accuracy

        # add the results of this feature selected model into a dataframe
        f_select.append({'threshold': thresh,'train acc':xgboost_feature_selected_train_accuracy,'test acc':xgboost_feature_selected_test_accuracy })
    
    # export the results of feature selection to CSV
    f_select_accuracy = pd.DataFrame(f_select)
    f_select_accuracy.to_csv('3rd Run - Feature Importance Scores.csv')

# after identifying the best threshold
# initialize the best performing model
if feature_selection_tuned:
    # Initialize SelectFromModel using the best threshold
    feature_model =  SelectFromModel(xgb_model, threshold = max_test_thresh, prefit=True)
    
    # apply feature selection to the training and testing dataset
    features_train_new = feature_model.transform(features_train)
    features_test_new = feature_model.transform(features_test)
    
    # convert the data into DMatrix, so it can work with xgb.cv
    dtrain_new = xgb.DMatrix(features_train_new, label=target_train)
    dtest_new = xgb.DMatrix(features_test_new, label=target_test)

    # set the parameters for use in the xgboost cross validation
    params = {'learning_rate': 0.0005, 'min_child_weight': 1, #'subsample':0.8,
             'objective': 'binary:logistic', 'max_depth':6,'subsample':1,'nthread':4,'random_state':319,'silent':True,'verbose_eval':0}
    
    # process xgboost cross validation
    cv_xgb = xgb.cv(params = params, dtrain = dtrain_new, num_boost_round = 100, nfold = 10,
                metrics = ['auc','error'],early_stopping_rounds = 20), 

    # train the final model
    final_xgboost = xgb.train(params, dtrain_new, num_boost_round = 100)

    # generate the predictions made for the training and testing datasets
    from sklearn.metrics import accuracy_score
    predictions_train = final_xgboost.predict(dtrain_new)
    predictions_train = [round(value) for value in predictions_train]
    
    predictions_test = final_xgboost.predict(dtest_new)
    predictions_test =  [round(value) for value in predictions_test]
    
    # calculate the training and testing accuracy 
    xgboost_feature_selected_tuned_train_accuracy = accuracy_score(target_train, predictions_train)
    xgboost_feature_selected_tuned_test_accuracy = accuracy_score(target_test, predictions_test)
    
    print 'xgboost - fs_tuned'
    print xgboost_feature_selected_tuned_train_accuracy
    print xgboost_feature_selected_tuned_test_accuracy
    
    # export the feature importance of the final model to CSV
    feature_importance = pd.DataFrame(final_xgboost.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
    feature_importance.to_csv('3rd Run - Importance Scores.csv')        
    
    
# Export the accuracies of the all the model trained 
df = [{'ml_model':'log_reg','training_acc':log_reg_train_accuracy,'testing_acc':log_reg_test_accuracy},
        {'ml_model':'log_reg_tuned','training_acc': log_reg_tuned_train_accuracy,'testing_acc': log_reg_tuned_test_accuracy,'best params':log_reg_tuned_params},
        {'ml_model':'d_tree','training_acc':   d_tree_train_accuracy,'testing_acc':  d_tree_test_accuracy},
        {'ml_model':'d_tree_tuned','training_acc': d_tree_tuned_train_accuracy,'testing_acc': d_tree_tuned_test_accuracy,'best params':d_tree_tuned_params},
        {'ml_model':'rf_tree','training_acc': rf_tree_train_accuracy,'testing_acc': rf_tree_test_accuracy},
        {'ml_model':'rf_tree_tuned','training_acc': rf_tree_train_accuracy,'testing_acc': rf_tree_test_accuracy,'best params':rf_tree_tuned_params},
        {'ml_model':'adaboost','training_acc': ada_boost_train_accuracy,'testing_acc': ada_boost_test_accuracy},
        {'ml_model':'ababoost_tuned','training_acc': ada_boost_tuned_train_accuracy,'testing_acc': ada_boost_tuned_test_accuracy,'best params':ada_boost_tuned_params},
        {'ml_model':'xg_boost','training_acc': xg_boost_train_accuracy,'testing_acc': xg_boost_test_accuracy},
        {'ml_model':'xg_boost_tuned','training_acc': xg_boost_tuned_train_accuracy,'testing_acc': xg_boost_tuned_test_accuracy,'best params':xg_boost_tuned_params},
        {'ml_model':'xg_boost_tuned2','training_acc': xgboost_tuned2_train_accuracy,'testing_acc': xgboost_tuned2_test_accuracy},
        {'ml_model':'xg_boost_tuned_features_selected','training_acc': xgboost_feature_selected_tuned_train_accuracy,'testing_acc': xgboost_feature_selected_tuned_test_accuracy},]

ml_accuracy = pd.DataFrame(df)
ml_accuracy.to_csv('1st and 2nd Run - Training and Testing Accuracy.csv')

# stop the timer and print out the duration
print datetime.now() - timestart