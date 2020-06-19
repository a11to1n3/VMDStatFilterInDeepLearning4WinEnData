#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:27:53 2019

@author: duypham
"""

from tensorflow import keras
import tensorflow as tf



def build(data_shape_1, data_shape_2):
    """
    This functions builds the desired LSTM model
    
    Parameters:
        data_shape (int): second dimension of the data
        
    Returns:
        model: LSTM architecture model with 2 layers and 20 units in each 
    """
    # create NN model    
    # design network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(20,kernel_initializer='glorot_normal', return_sequences=True, input_shape=(data_shape_1,data_shape_2)))
    model.add(keras.layers.LSTM(20,kernel_initializer='glorot_normal', activation='selu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(data_shape_1,kernel_initializer='glorot_normal'))
    
    return model