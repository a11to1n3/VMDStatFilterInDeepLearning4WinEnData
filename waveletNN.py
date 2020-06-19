#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:16:17 2020

@author: duypham
"""

import pywt

def waveletFunc(signal, wavelet="db2"):
    coeff = pywt.wavedec(signal, wavelet, level=1)
    cA1, cD1 = coeff
    cD11 = np.zeros_like(cD1)
    cA11 = np.zeros_like(cA1)    
    coeff1 = list([cA1,cD11])
    coeff2 = list([cA11,cD1])
    highCancelledSignal = pywt.waverec(coeff1, wavelet)
    lowCancelledSignal = pywt.waverec(coeff2, wavelet)
    return highCancelledSignal, lowCancelledSignal

def extract(data_load):
    high_end = np.zeros_like(data_load)
    low_
    high_end, low = lowfreq_cancel(data_l)