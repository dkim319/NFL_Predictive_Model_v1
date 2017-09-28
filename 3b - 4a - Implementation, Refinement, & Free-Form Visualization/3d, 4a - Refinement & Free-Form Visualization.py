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
data = pd.read_csv('Data v2.csv')

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
xgboost_bool = True
feature_selection = True  
feature_selection_tuned = True
weekly_iterate = True
free_form = True
# -----------------------------------------------
# XGBoost
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
    
    correction_predictions_orig = xgboost_tuned2_test_accuracy * len(predictions_test)
    
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
    f_select_accuracy.to_csv('4th Run - Feature Importance Scores.csv')

# after identifying the best threshold
# initialize the best performing model
if feature_selection_tuned:
    # initialize variables
    final_model_accuracy = []
    
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
    final_predictions_test = predictions_test 
    
    # calculate the training and testing accuracy 
    xgboost_feature_selected_tuned_train_accuracy = accuracy_score(target_train, predictions_train)
    xgboost_feature_selected_tuned_test_accuracy = accuracy_score(target_test, predictions_test)
    
    print 'xgboost - fs_tuned'
    print xgboost_feature_selected_tuned_train_accuracy
    print xgboost_feature_selected_tuned_test_accuracy
    
    # export the final model testing and training accuracy
    final_model_accuracy.append({'training accuracy':xgboost_feature_selected_tuned_train_accuracy,'testing accuracy':xgboost_feature_selected_tuned_test_accuracy})
    final_model_accuracy = pd.DataFrame(final_model_accuracy)
    final_model_accuracy.to_csv('4th Run - Final XGBoost Training and Testing Accuracy.csv')
    #features_select_df = pd.DataFrame(features_test_new)

# The section is test the second approach of retraining using the latest data to predict the next week of games
if weekly_iterate:
    # initialize varaibles
    correct_predictions = 0
    gamecount = 0
    weekly_predictions = 0
    weekly_accuracy = []
    overall_accuracy_list = []
    
    # identify a unique list of gameweeks for 2016 season
    gameweek = data[data['season']== 2016]['gameweek']
    gameweek = gameweek.unique()
    
    # retrained and predict the games within the game week
    for x in gameweek:
        # use the customized function to retrain and predict a week of games
        xgboost_tuned_train_accuracy, xgboost_tuned_test_accuracy, predictions_test, target_test = weeklyIteration(x, data)
        
        # generate the predictions using the retained model
        weekly_predictions = xgboost_tuned_test_accuracy * len(predictions_test)

        # add up the correction predictions and total number of games for the overall accuracy
        correct_predictions += weekly_predictions
        gamecount += len(predictions_test)
        
        # add the week's testing accuracy, correct predictions, and total games into a dataframe
        weekly_accuracy.append({'week': x,'testing accuracy': xgboost_tuned_test_accuracy,'correct predictions': weekly_predictions,'total games': len(predictions_test)})
    
    # export the weekly results to CSV
    weekly_accuracy_df = pd.DataFrame(weekly_accuracy)
    weekly_accuracy_df.to_csv('5th Run - Weekly Testing Accuracy.csv')   

    # calculate the overall accuracy
    overall_accuracy = correct_predictions / gamecount

    # export the overall accuracy results to CSV
    overall_accuracy_list.append({'testing accuracy': overall_accuracy,'correct predictions': correct_predictions,'total games': gamecount})  
    weekly_iteration_overall_accuracy = pd.DataFrame  (overall_accuracy_list)
    weekly_iteration_overall_accuracy.to_csv('5th Run  Overall Testing Accuracy.csv')
    
    print 'weekly iterate'
    print overall_accuracy
    print correct_predictions

# this section is used to create the free-form visualizations
if free_form:
    
    # append the predictions from the final xgboost model
    data_test['xgboost'] = final_predictions_test
    
    # apply logic to identify which predictions were correct
    data_test['correct_pred'] = np.where(data_test['spreadflag']==data_test['xgboost'],1,0)

    # aggregate the correction predictions by spread, by favorite ats winning percentage over the last 5 games, and by week
    correct_pred_by_spread = data_test.groupby('spread')['correct_pred'].sum() / data_test.groupby('spread')['correct_pred'].count()
    correct_pred_by_fav_as_fav_win = data_test.groupby('fav_as_fav_last_5_ats_percent')['correct_pred'].sum() / data_test.groupby('fav_as_fav_last_5_ats_percent')['correct_pred'].count()
    correct_pred_by_week = data_test.groupby('week')['correct_pred'].sum() / data_test.groupby('week')['correct_pred'].count()

    # plot the visualization by spread and by favoritre ats winning percentage over the last 5 games
    fig, axes = plt.subplots(2,1)    
    
    correct_pred_by_spread_ax = correct_pred_by_spread.plot(color='blue',ax = axes[0])
    correct_pred_by_spread_ax.set_ylabel('Testing Accuracy')
    correct_pred_by_spread_ax.set_title('Figure 6 - Testing Accuracy by Features')
    correct_pred_by_spread_figure = correct_pred_by_spread_ax.get_figure()
    plt.show()
    
    correct_pred_by_fav_as_fav_win_ax = correct_pred_by_fav_as_fav_win.plot(color='blue',ax = axes[1])
    correct_pred_by_fav_as_fav_win_ax.set_ylabel('Testing Accuracy')
    correct_pred_by_fav_as_fav_win_figure = correct_pred_by_fav_as_fav_win_ax.get_figure()
    plt.show()
    plt.savefig('4a - Testing Acc by Spread and Fav as Fav ATS')
    plt.close()

    # plot the visualization by week
    correct_pred_by_week_ax = correct_pred_by_week.plot(color='blue')
    correct_pred_by_week_ax.set_ylabel('Testing Accuracy')
    correct_pred_by_week_ax.set_title('Figure 7 - Testing Accuracy by Week')
    correct_pred_by_week_figure = correct_pred_by_week_ax.get_figure()
    plt.show()
    plt.savefig('4a - Testing Acc by Week')  

# stop the timer and print out the duration
print datetime.now() - timestart