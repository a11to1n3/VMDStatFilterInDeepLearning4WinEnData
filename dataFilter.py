#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:49:16 2019

@author: duypham
"""

from numpy import mean
from numpy import std

def filterWithConfidenceLevel(data_all,confidence_level):
    """Summary or Description of the Function
    this function filters the data by a given confidence level.
    
    Parameters:
    data_all (array): the data about to be filtered
    confidence_level (int): given confidence level ranging from 90 to 99.99932
        
    Returns:
    data_all (array): the filtered data
    """
    
    if confidence_level == 90:
        sigma = 1.645
    elif confidence_level == 91:
        sigma = 1.695
    elif confidence_level == 92:
        sigma = 1.75
    elif confidence_level == 93:
        sigma = 1.81
    elif confidence_level == 94:
        sigma = 1.88
    elif confidence_level == 95:
        sigma = 1.96
    elif confidence_level == 96:
        sigma = 2.05
    elif confidence_level == 97:
        sigma = 2.17
    elif confidence_level == 98:
        sigma = 2.33
    elif confidence_level == 99:
        sigma = 2.58
    elif confidence_level == 99.73:
        sigma = 3
    elif confidence_level == 99.99366:
        sigma = 4
    elif confidence_level == 99.99932:
        sigma = 4.5
    
    
    
    # calculate summary statistics
    data_mean, data_std = mean(data_all[:,:,0].reshape(-1)), std(data_all[:,:,0].reshape(-1))
    #data_min = min(data_all[:,:,0].reshape(-1))
    #data_max = max(data_all[:,:,0].reshape(-1))
    # identify outliers
    cut_off = data_std * sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off
    
    for i in range(data_all.shape[0]):
        for j in range(data_all.shape[1]):
            if data_all[i,j,0] < lower:
                data_all[i,j,0]= lower
            elif data_all[i,j,0] > upper:
                data_all[i,j,0] = upper
                
                
    return data_all