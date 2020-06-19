#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:22:45 2020

@author: duypham
"""

import pandas as pd
import numpy as np


load_data = pd.read_csv('NYISO.csv')
count = 0
prev = 0
for i in range(1,len(load_data)):
    if load_data.values[i,0] != load_data.values[i-1,0]:
        count+=1
        print('day no.',count,' ')
        print(load_data.values[i,0])
        print(' number of regions ', i-prev)
        prev = i