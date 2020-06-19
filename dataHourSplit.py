#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:22:00 2019

@author: duypham
"""

def splitToDifferentTimeSpan(daySpan, dayType):
    """Summary or Description of the Function
    this function splits the data into 4 timespan as in the paper.
    
    Parameters:
    daySpan (array): whole day array of either Monday or Sunday with 133 elements 

    Returns:
    daySpan 1 (array): first timespan
    daySpan 2 (array): second timespan
    daySpan 3 (array): third timespan
    daySpan 4 (array): fourth timespan
    """
    
    if dayType == 'Mon':
        # if it is Monday, split to 4 timespans 0h - 4h, 5h - 6h, 7h - 16h and 17h - 23h
        return daySpan[:,:5,:], daySpan[:,5:7,:], daySpan[:,7:17,:], daySpan[:,17:,:]
    else:
        # if it is Sunday, split to 4 timespans 0h - 4h, 5h - 6h, 7h - 15h and 16h - 23h
        return daySpan[:,:5,:], daySpan[:,5:7,:], daySpan[:,7:16,:], daySpan[:,16:,:]