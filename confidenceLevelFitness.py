#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:27:53 2019

@author: duypham
"""

from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Activation, Dropout
from keras import Sequential
import numpy as np

def calculateMAPE(data, original_data, EPOCHS=100, BATCH_SIZE=2):
    
    # Split train, test sets
    inp = data[:-int(0.1*len(data))-1]
    out = data[1:-int(0.1*len(data))]
    test_inp = original_data[-int(0.1*len(data)):-1]
    test_out = original_data[-int(0.1*len(data))+1:]
    x_train, x_val, y_train, y_val = train_test_split(inp,out,
                                                      random_state=42, test_size =0.33,shuffle=False)
    
    # create NN model    
    # design network
    model = Sequential()
    model.add(Dense(100, input_shape=(data.shape[1],133)))
    model.add(Activation('selu'))
    model.add(Dense(100))
    model.add(Activation('selu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(data.shape[1]))
    
    model.compile(loss='mae', optimizer='adam', metrics=['mse','mape'])
    # fit network
    
    history = model.fit(x_train, y_train[:,:,0], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val[:,:,0]), verbose=1, shuffle=True)
    # calculate MAPE
    AVG = []
    for i in range(len(test_inp)):
        test_pred = model.predict(test_inp[i].reshape(-1,data.shape[1],133))
        AVG.append(abs((np.e**test_pred.T.reshape(-1)-np.e**test_out[i,:,0])*100/np.e**test_pred.T.reshape(-1)))
    AVG = np.array(AVG)
    return np.mean(AVG)