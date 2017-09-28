# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:21:23 2017

@author: DKIM
"""

# required libraries loaded 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('Data.csv')

# replace any null values with 0
data = data.fillna(0)

# the booleans are used to control which plot is generated
spread_vis = True
total_vis = True
fav_last_5_percent_vis = True
und_last_5_percent_vis = True

# this visualization displays the favorite win % and count by spread
if spread_vis:
    spread_agg =  data.groupby(['spread'])['spreadflag'].mean()#.sum()/ data.groupby(['spread'])['spreadflag'].count()
    spread_count = data.groupby(['spread'])['spreadflag'].count()
    
    fig, axes = plt.subplots(2,1)
    
    spread_agg_ax = spread_agg.plot(ax = axes[0])
    spread_agg_ax.set_ylabel('favorite win %')
    spread_agg_ax.set_title('Figure 1 - Spread')
    spread_agg_figure = spread_agg_ax.get_figure()
    plt.show()
    #plt.clf()
    
    spread_count_ax = spread_count.plot(kind = 'line',ax = axes[1])
    spread_count_ax.set_ylabel('favorite win count')
    spread_count_figure = spread_count_ax.get_figure()
    plt.show()
    plt.savefig('2b - fig 1 - spread_vis.png')
    #plt.close()

# this visualization displays the favorite win % and count by total
if total_vis:
    total_agg =  data.groupby(['total'])['spreadflag'].mean()#.sum()/ data.groupby(['spread'])['spreadflag'].count()
    total_count = data.groupby(['total'])['spreadflag'].count()
     
    fig, axes = plt.subplots(2,1)

    total_agg_ax = total_agg.plot(ax = axes[0])
    total_agg_ax.set_ylabel('favorite win %')
    total_agg_ax.set_title('Figure 2 - Total')
    total_agg_figure = total_agg_ax.get_figure()
    plt.show()
    #plt.clf()
    
    total_count_ax = total_count.plot(kind = 'line',ax = axes[1])
    total_count_ax.set_ylabel('favorite win count')
    total_count_figure = total_count_ax.get_figure()
    plt.show()
    plt.savefig('2b - fig 2 - total_vis.png')
    #plt.close()

# this visualization displays the favorite win % and count by favorite winning percent over the last 5 games
if fav_last_5_percent_vis:
    fav_last_5_percent_vis_agg =  data.groupby(['fav_last_5_percent'])['spreadflag'].mean()#.sum()/ data.groupby(['spread'])['spreadflag'].count()
    fav_last_5_percent_vis_count = data.groupby(['fav_last_5_percent'])['spreadflag'].count()
     
    fig, axes = plt.subplots(2,1)

    fav_last_5_percent_vis_agg_ax = fav_last_5_percent_vis_agg.plot(ax = axes[0])
    fav_last_5_percent_vis_agg_ax.set_ylabel('favorite win %')
    fav_last_5_percent_vis_agg_ax.set_title('Figure 3 - Fav Win % Last 5')
    fav_last_5_percent_vis_agg_figure = fav_last_5_percent_vis_agg_ax.get_figure()
    plt.show()
    #plt.clf()
    
    fav_last_5_percent_vis_count_ax = fav_last_5_percent_vis_count.plot(kind = 'line',ax = axes[1])
    fav_last_5_percent_vis_count_ax.set_ylabel('favorite win count')
    fav_last_5_percent_vis_count_figure = fav_last_5_percent_vis_count_ax.get_figure()
    plt.show()
    plt.savefig('2b - fig 3 - fav_last_5_percent.png')
    #plt.close()

# this visualization displays the favorite win % and count by underdog winning percent over the last 5 games
if und_last_5_percent_vis:
    undlast_5_percent_vis_agg =  data.groupby(['und_last_5_percent'])['spreadflag'].mean()#.sum()/ data.groupby(['spread'])['spreadflag'].count()
    und_last_5_percent_vis_count = data.groupby(['und_last_5_percent'])['spreadflag'].count()
     
    fig, axes = plt.subplots(2,1)

    und_last_5_percent_vis_agg_ax = undlast_5_percent_vis_agg.plot(ax = axes[0])
    und_last_5_percent_vis_agg_ax.set_ylabel('favorite win %')
    und_last_5_percent_vis_agg_ax.set_title('Figure 4 - Und Win % Last 5')
    und_last_5_percent_vis_agg_figure = und_last_5_percent_vis_agg_ax.get_figure()
    plt.show()
    #plt.clf()
    
    und_last_5_percent_vis_count_ax = und_last_5_percent_vis_count.plot(kind = 'line',ax = axes[1])
    und_last_5_percent_vis_count_ax.set_ylabel('favorite win count')
    und_last_5_percent_vis_count_figure = und_last_5_percent_vis_count_ax.get_figure()
    plt.show()
    plt.savefig('2b - fig 4 - und_last_5_percent.png')
    #plt.close()



