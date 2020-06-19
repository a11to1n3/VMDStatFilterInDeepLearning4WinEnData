#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:27:53 2019

@author: duypham
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
import numpy as np

def build(data_shape):
    """
    This functions builds the desired ANN model
    
    Parameters:
        data_shape (int): second dimension of the data
        
    Returns:
        model: ANN architecture model
    """
    # create NN model    
    # design network
    model = Sequential()
    model.add(Dense(100, input_shape=(data_shape,133)))
    model.add(Activation('selu'))
    model.add(Dense(100))
    model.add(Activation('selu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(data_shape))
    
    return model