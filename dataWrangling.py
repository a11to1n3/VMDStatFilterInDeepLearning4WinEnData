#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:49:16 2019

@author: duypham
"""

import pandas as pd
import numpy as np

def csvToArray(filename):
    """Summary or Description of the Function

    Parameters:
    filenam (str): the name of input file (with csv extention)

    Returns:
    data (array): returning arrays after wrangling to 133 elements

    """
    # %% Load data
    df = pd.read_csv(filename)
    
    # %% Create array
    data = np.zeros((len(df),24,133))
    #data = np.zeros((3299,24,133))
    #Set holiday variable
    data[:,0,-1] = df.values[:,-2]
    data[:,1,-1] = df.values[:,-2]
    data[:,2,-1] = df.values[:,-2]
    data[:,3,-1] = df.values[:,-2]
    data[:,4,-1] = df.values[:,-2]
    data[:,5,-1] = df.values[:,-2]
    data[:,6,-1] = df.values[:,-2]
    data[:,7,-1] = df.values[:,-2]
    data[:,8,-1] = df.values[:,-2]
    data[:,9,-1] = df.values[:,-2]
    data[:,10,-1] = df.values[:,-2]
    data[:,11,-1] = df.values[:,-2]
    data[:,12,-1] = df.values[:,-2]
    data[:,13,-1] = df.values[:,-2]
    data[:,14,-1] = df.values[:,-2]
    data[:,15,-1] = df.values[:,-2]
    data[:,16,-1] = df.values[:,-2]
    data[:,17,-1] = df.values[:,-2]
    data[:,18,-1] = df.values[:,-2]
    data[:,19,-1] = df.values[:,-2]
    data[:,20,-1] = df.values[:,-2]
    data[:,21,-1] = df.values[:,-2]
    data[:,22,-1] = df.values[:,-2]
    data[:,23,-1] = df.values[:,-2]
    #Set hour variable
    data[:,0,1] = 1
    data[:,1,2] = 1
    data[:,2,3] = 1
    data[:,3,4] = 1
    data[:,4,5] = 1
    data[:,5,6] = 1
    data[:,6,7] = 1
    data[:,7,8] = 1
    data[:,8,9] = 1
    data[:,9,10] = 1
    data[:,10,11] = 1
    data[:,11,12] = 1
    data[:,12,13] = 1
    data[:,13,14] = 1
    data[:,14,15] = 1
    data[:,15,16] = 1
    data[:,16,17] = 1
    data[:,17,18] = 1
    data[:,18,19] = 1
    data[:,19,20] = 1
    data[:,20,21] = 1
    data[:,21,22] = 1
    data[:,22,23] = 1
    data[:,23,24] = 1
    #Set hourly values
    data[:,0,0] = df.values[:,2]
    data[:,1,0] = df.values[:,3]
    data[:,2,0] = df.values[:,4]
    data[:,3,0] = df.values[:,5]
    data[:,4,0] = df.values[:,6]
    data[:,5,0] = df.values[:,7]
    data[:,6,0] = df.values[:,8]
    data[:,7,0] = df.values[:,9]
    data[:,8,0] = df.values[:,10]
    data[:,9,0] = df.values[:,11]
    data[:,10,0] = df.values[:,12]
    data[:,11,0] = df.values[:,13]
    data[:,12,0] = df.values[:,14]
    data[:,13,0] = df.values[:,15]
    data[:,14,0] = df.values[:,16]
    data[:,15,0] = df.values[:,17]
    data[:,16,0] = df.values[:,18]
    data[:,17,0] = df.values[:,19]
    data[:,18,0] = df.values[:,20]
    data[:,19,0] = df.values[:,21]
    data[:,20,0] = df.values[:,22]
    data[:,21,0] = df.values[:,23]
    data[:,22,0] = df.values[:,24]
    #for i in range(len(df)):
    #    data[i,21,0] = float(df.values[i,23])
    #    data[i,22,0] = float(df.values[i,24])
    data[:,23,0] = df.values[:,25]
    #Set weekday & day indices
    for i in range(len(df)):
        if df.values[i,0] == 'CN':
            data[i,0,31] = 1
            data[i,1,31] = 1
            data[i,2,31] = 1
            data[i,3,31] = 1
            data[i,4,31] = 1
            data[i,5,31] = 1
            data[i,6,31] = 1
            data[i,7,31] = 1
            data[i,8,31] = 1
            data[i,9,31] = 1
            data[i,10,31] = 1
            data[i,11,31] = 1
            data[i,12,31] = 1
            data[i,13,31] = 1
            data[i,14,31] = 1
            data[i,15,31] = 1
            data[i,16,31] = 1
            data[i,17,31] = 1
            data[i,18,31] = 1
            data[i,19,31] = 1
            data[i,20,31] = 1
            data[i,21,31] = 1
            data[i,22,31] = 1
            data[i,23,31] = 1
        else:
            a = int(df.values[i,0])+23
            data[i,0,a] = 1
            data[i,1,a] = 1
            data[i,2,a] = 1
            data[i,3,a] = 1
            data[i,4,a] = 1
            data[i,5,a] = 1
            data[i,6,a] = 1
            data[i,7,a] = 1
            data[i,8,a] = 1
            data[i,9,a] = 1
            data[i,10,a] = 1
            data[i,11,a] = 1
            data[i,12,a] = 1
            data[i,13,a] = 1
            data[i,14,a] = 1
            data[i,15,a] = 1
            data[i,16,a] = 1
            data[i,17,a] = 1
            data[i,18,a] = 1
            data[i,19,a] = 1
            data[i,20,a] = 1
            data[i,21,a] = 1
            data[i,22,a] = 1
            data[i,23,a] = 1
        b = int(df.values[i,1].split('/')[1])+31
        data[i,0,b] = 1
        data[i,1,b] = 1
        data[i,2,b] = 1
        data[i,3,b] = 1
        data[i,4,b] = 1
        data[i,5,b] = 1
        data[i,6,b] = 1
        data[i,7,b] = 1
        data[i,8,b] = 1
        data[i,9,b] = 1
        data[i,10,b] = 1
        data[i,11,b] = 1
        data[i,12,b] = 1
        data[i,13,b] = 1
        data[i,14,b] = 1
        data[i,15,b] = 1
        data[i,16,b] = 1
        data[i,17,b] = 1
        data[i,18,b] = 1
        data[i,19,b] = 1
        data[i,20,b] = 1
        data[i,21,b] = 1
        data[i,22,b] = 1
        data[i,23,b] = 1
        c = int(df.values[i,-1])+62
        data[i,0,c] = 1
        data[i,1,c] = 1
        data[i,2,c] = 1
        data[i,3,c] = 1
        data[i,4,c] = 1
        data[i,5,c] = 1
        data[i,6,c] = 1
        data[i,7,c] = 1
        data[i,8,c] = 1
        data[i,9,c] = 1
        data[i,10,c] = 1
        data[i,11,c] = 1
        data[i,12,c] = 1
        data[i,13,c] = 1
        data[i,14,c] = 1
        data[i,15,c] = 1
        data[i,16,c] = 1
        data[i,17,c] = 1
        data[i,18,c] = 1
        data[i,19,c] = 1
        data[i,20,c] = 1
        data[i,21,c] = 1
        data[i,22,c] = 1
        data[i,23,c] = 1
        d = int(df.values[i,1].split('/')[0])+31
        data[i,0,d] = 1
        data[i,1,d] = 1
        data[i,2,d] = 1
        data[i,3,d] = 1
        data[i,4,d] = 1
        data[i,5,d] = 1
        data[i,6,d] = 1
        data[i,7,d] = 1
        data[i,8,d] = 1
        data[i,9,d] = 1
        data[i,10,d] = 1
        data[i,11,d] = 1
        data[i,12,d] = 1
        data[i,13,d] = 1
        data[i,14,d] = 1
        data[i,15,d] = 1
        data[i,16,d] = 1
        data[i,17,d] = 1
        data[i,18,d] = 1
        data[i,19,d] = 1
        data[i,20,d] = 1
        data[i,21,d] = 1
        data[i,22,d] = 1
        data[i,23,d] = 1
        if d-115 in [1,2,3]:
            data[i,0,128] = 1
            data[i,1,128] = 1
            data[i,2,128] = 1
            data[i,3,128] = 1
            data[i,4,128] = 1
            data[i,5,128] = 1
            data[i,6,128] = 1
            data[i,7,128] = 1
            data[i,8,128] = 1
            data[i,9,128] = 1
            data[i,10,128] = 1
            data[i,11,128] = 1
            data[i,12,128] = 1
            data[i,13,128] = 1
            data[i,14,128] = 1
            data[i,15,128] = 1
            data[i,16,128] = 1
            data[i,17,128] = 1
            data[i,18,128] = 1
            data[i,19,128] = 1
            data[i,20,128] = 1
            data[i,21,128] = 1
            data[i,22,128] = 1
            data[i,23,128] = 1
        elif d-115 in [4,5,6]:
            data[i,0,129] = 1
            data[i,1,129] = 1
            data[i,2,129] = 1
            data[i,3,129] = 1
            data[i,4,129] = 1
            data[i,5,129] = 1
            data[i,6,129] = 1
            data[i,7,129] = 1
            data[i,8,129] = 1
            data[i,9,129] = 1
            data[i,10,129] = 1
            data[i,11,129] = 1
            data[i,12,129] = 1
            data[i,13,129] = 1
            data[i,14,129] = 1
            data[i,15,129] = 1
            data[i,16,129] = 1
            data[i,17,129] = 1
            data[i,18,129] = 1
            data[i,19,129] = 1
            data[i,20,129] = 1
            data[i,21,129] = 1
            data[i,22,129] = 1
            data[i,23,129] = 1
        elif d-115 in [7,8,9]:
            data[i,0,130] = 1
            data[i,1,130] = 1
            data[i,2,130] = 1
            data[i,3,130] = 1
            data[i,4,130] = 1
            data[i,5,130] = 1
            data[i,6,130] = 1
            data[i,7,130] = 1
            data[i,8,130] = 1
            data[i,9,130] = 1
            data[i,10,130] = 1
            data[i,11,130] = 1
            data[i,12,130] = 1
            data[i,13,130] = 1
            data[i,14,130] = 1
            data[i,15,130] = 1
            data[i,16,130] = 1
            data[i,17,130] = 1
            data[i,18,130] = 1
            data[i,19,130] = 1
            data[i,20,130] = 1
            data[i,21,130] = 1
            data[i,22,130] = 1
            data[i,23,130] = 1
        else:
            data[i,0,131] = 1
            data[i,1,131] = 1
            data[i,2,131] = 1
            data[i,3,131] = 1
            data[i,4,131] = 1
            data[i,5,131] = 1
            data[i,6,131] = 1
            data[i,7,131] = 1
            data[i,8,131] = 1
            data[i,9,131] = 1
            data[i,10,131] = 1
            data[i,11,131] = 1
            data[i,12,131] = 1
            data[i,13,131] = 1
            data[i,14,131] = 1
            data[i,15,131] = 1
            data[i,16,131] = 1
            data[i,17,131] = 1
            data[i,18,131] = 1
            data[i,19,131] = 1
            data[i,20,131] = 1
            data[i,21,131] = 1
            data[i,22,131] = 1
            data[i,23,131] = 1
    return data