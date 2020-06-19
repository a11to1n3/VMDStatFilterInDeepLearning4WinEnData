#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:27:53 2019

@author: duypham
"""

from tensorflow import keras
from tensorflow.keras import layers

def build(data_shape_1, data_shape_2):
    """
    This functions builds the desired LSTM model
    
    Parameters:
        data_shape (int): second dimension of the data
        
    Returns:
        model: CNN-LSTM architecture model with 2 layers and 20 units in each 
    """
    # create NN model    
    # design network
    
    inputs = keras.Input(shape=(data_shape_1, data_shape_2), name='inp')
    cnn1 = layers.Conv1D(16, 5, activation='relu')(inputs)
    cnn2 = layers.Conv1D(32, 3, activation='relu')(cnn1)
    cnn3 = layers.Conv1D(64, 3, activation='relu')(cnn2)
    cnn3 = layers.Flatten()(cnn3)
    lstm = layers.LSTM(100,return_sequences = True, activation='relu')(inputs)
    lstm = layers.Flatten()(lstm)
    x = layers.concatenate([cnn3,lstm])
    x = layers.Dense(100, activation='sigmoid')(x)
    outputs = layers.Dense(24)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    
    return model