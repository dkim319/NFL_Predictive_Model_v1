# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:21:23 2017

@author: DKIM
"""

import pandas as pd
import numpy as np

seednumber = 319

data = pd.read_csv('Data.csv')

data = data.fillna(0)

statistics = data.describe()
statistics.to_csv('stats.csv')