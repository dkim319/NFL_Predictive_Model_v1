# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:33:56 2017

@author: DKIM
"""

# required libraries loaded
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data.csv')

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

# subset the training data for analysis purposes
analysis_data = data[data['season'] <= 2015]
analysis_data.reset_index()

# the purpose of this aggregation is to calculate the performance of a naive approach of selecting the favorite or underdog for every game
# aggregate the data by favorite wins and underdog wins for plotting
fav_win_avg_by_season = analysis_data.groupby('season')['spreadflag'].sum() / analysis_data.groupby('season')['spreadflag'].count()
und_win_avg_by_season = (analysis_data.groupby('season')['spreadflag'].count() - analysis_data.groupby('season')['spreadflag'].sum()) / analysis_data.groupby('season')['spreadflag'].count()

fig, axes = plt.subplots(2,1)

fav_win_avg_by_season_ax = fav_win_avg_by_season.plot(kind = 'bar',ax = axes[0])
fav_win_avg_by_season_ax.set_ylabel('favorite win %')
fav_win_avg_by_season_ax.set_title('Figure 5 - Winning % by Season')
fav_win_avg_by_season_figure = fav_win_avg_by_season_ax.get_figure()
plt.show()

und_win_avg_by_season_ax = und_win_avg_by_season.plot(kind = 'bar',ax = axes[1])
und_win_avg_by_season_ax.set_ylabel('underdog win %')
und_win_avg_by_season_figure = und_win_avg_by_season_ax.get_figure()
plt.show()
plt.savefig('3b - fig 5 - benchmark - win % by season.png')
#plt.close()

# print out overall averages by favorite and underdog
print 'Favorite Winning % by Season'
print fav_win_avg_by_season
print 'Overall Favorite Winning %'
print fav_win_avg_by_season.mean()
print ''
print 'Underdog Winning % by Season'
print und_win_avg_by_season
print 'Overall Underdog Winning %'
print und_win_avg_by_season.mean()