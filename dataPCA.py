#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:59:45 2019

@author: duytaiba
"""

# %% Imports


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def plotPCAandSplit(data_all):
    
    """Summary or Description of the Function
    this function scale the data by taking log of the original data 
    and differencing the two consecutive values. 
    
    Parameters:
    data_all (array): total array with 133 elements 

    Returns:
    mon2 (array): scaled data of all the mondays from 01.01.2014 to 23.12.2018
    sun2 (array): scaled data of all the sundays from 01.01.2014 to 23.12.2018
    rests2 (array): scaled data of all the remaing days from 01.01.2014 to 23.12.2018
    mon_orig (array): raw data of all the mondays from 01.01.2014 to 23.12.2018
    sun_orig (array): raw data of all the sunndays from 01.01.2014 to 23.12.2018
    rests_orig (array): raw data of all the remaining days from 01.01.2014 to 23.12.2018
    presun_orig (array): raw data of all the day before sundays from 01.01.2014 to 23.12.2018
    premon_orig (array): raw data of all the day before mondays from 01.01.2014 to 23.12.2018

    """
    
    # Differencing
        
    df_data_all = pd.Series(data_all[:,:,0].flatten())
    ts_log = np.log(df_data_all)
    #print(ts_log.isna().sum())
    diff = ts_log - ts_log.shift(24)
    #print(diff.isna().sum())
    #print(np.argwhere(np.isinf(diff)))
    #print(np.argwhere(np.isinf(diff.dropna().drop(22405).drop(22429).drop(22404))))
    diff_drop = np.array(diff[27768:])
    
    # Plot PDF (histogram)
    
    # Cut from 01.01.2014
    diff_main = np.zeros((data_all.shape[0] - 27768//24,24,133),dtype=np.float64)
    diff_main[:,:,0:1] = diff_drop.reshape(-1,24,1)
    diff_main[:,:,1:] = data_all[1157:,:,1:]
    np.savez('diff_whole.npz',diff_main)
    plt.hist(diff_drop,bins='sqrt',histtype = 'step',color='r')
    plt.title("PDF of the Difference Series")
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.show()
    plt.plot(diff_drop)
    plt.show()
    plt.boxplot(diff_drop)
    plt.show()
    
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(diff_main[:,:,0].reshape(-1,24))
    mon=[]
    mon1=[]
    mon2=[]
    sun=[]
    sun1=[]
    sun2=[]
    rests= []
    rests1=[]
    rests2=[]
    mon_orig=[]
    sun_orig=[]
    rests_orig=[]
    presun_orig = []
    premon_orig = []
    for i in range(principalComponents.shape[0]):
        if diff_main[i,0,25] == 1:
            mon.append(principalComponents[i])
            mon1.append(diff_main[i,:,0])
            mon2.append(diff_main[i])
            mon_orig.append(data_all[1157+i,:,:])
            premon_orig.append(data_all[1157+i-1,:,:])
            #plt.scatter(principalComponents[i,0],principalComponents[i,1],color='r',label="Monday")
        elif diff_main[i,0,31] == 1:
            sun.append(principalComponents[i])
            sun1.append(diff_main[i,:,0])
            sun2.append(diff_main[i])
            sun_orig.append(data_all[1157+i,:,:])
            presun_orig.append(data_all[1157+i-1,:,:])
            #plt.scatter(principalComponents[i,0],principalComponents[i,1],color='g',label="Sunday")
        else:
            #if diff_main[i,0,30] == 1:
                #sat_orig.append(data_all[1157+i,:,0])
            rests.append(principalComponents[i])
            rests1.append(diff_main[i,:,0])
            rests2.append(diff_main[i])
            rests_orig.append(data_all[1157+i-1,:,])
            #plt.scatter(principalComponents[i,0],principalComponents[i,1],color='black',label="The rests")
    mon = np.array(mon)
    sun = np.array(sun)
    rests = np.array(rests)
    mon1 = np.array(mon1)
    sun1 = np.array(sun1)
    rests1 = np.array(rests1)
    mon2 = np.array(mon2)
    sun2 = np.array(sun2)
    rests2 = np.array(rests2)
    mon_orig = np.array(mon_orig)
    sun_orig = np.array(sun_orig)
    rests_orig = np.array(rests_orig)
    presun_orig = np.array(presun_orig)
    premon_orig = np.array(premon_orig)
    
    
    # Plot PCA
    plt.scatter(mon[:,0],mon[:,1],marker='o',label="Change from Sundays to Mondays")
    plt.scatter(sun[:,0],sun[:,1],marker='v',label="Change from Saturdays to Sundays")
    plt.scatter(rests[:,0],rests[:,1],marker='s',label="The rests")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Principal Components Analysis")
    plt.legend()
    plt.show()
    
    return mon2, sun2, rests2, mon_orig, sun_orig, rests_orig, presun_orig, premon_orig

def diffandScaleISONE(data_all):
    
    """Summary or Description of the Function
    this function scale the data by taking log of the original data 
    and differencing the two consecutive values. 
    
    Parameters:
    data_all (array): total array with 133 elements 
    Returns:
    diff_main (array): scaled data of all the days from 01.01.2014 to 23.12.2018
    data_all (array): origin data of all the days from 01.01.2014 to 23.12.2018
    """
    
    
    # Differencing
        
    df_data_all = pd.Series(data_all[:,:,0].flatten())
    ts_log = np.log(df_data_all)
    #print(ts_log.isna().sum())
    diff = ts_log - ts_log.shift(24)
    #print(diff.isna().sum())
    #print(np.argwhere(np.isinf(diff)))
    #print(np.argwhere(np.isinf(diff.dropna().drop(22405).drop(22429).drop(22404))))
    diff_drop = np.array(diff[24:])
    
    # Plot PDF (histogram)
    
    # Cut from 01.01.2014
    diff_main = np.zeros((data_all.shape[0]-1,24,133),dtype=np.float64)
    diff_main[:,:,0:1] = diff_drop.reshape(-1,24,1)
    diff_main[:,:,1:] = data_all[1:,:,1:]
    np.savez('diff_whole.npz',diff_main)
    plt.hist(diff_drop,bins='sqrt',histtype = 'step',color='r')
    plt.title("PDF of the Difference Series")
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.show()
    plt.plot(diff_drop)
    plt.show()
    plt.boxplot(diff_drop)
    plt.show()

    
    return diff_main, data_all

def diffandScaleVMD(data_all):
    
    """Summary or Description of the Function
    this function scale the data by taking log of the original data 
    and differencing the two consecutive values. 
    
    Parameters:
    data_all (array): total array with 133 elements 

    Returns:
    diff_main (array): scaled data of all the days from 01.01.2014 to 23.12.2018
    data_all (array): origin data of all the days from 01.01.2014 to 23.12.2018
    """
    
    
    # Differencing
        
    df_data_all = pd.Series(data_all[:,:,0].flatten())
    #ts_log = np.log(df_data_all)
    #print(ts_log.isna().sum())
    diff = df_data_all - df_data_all.shift(24)
    diff_drop = diff[24:]
    
    #print(diff.isna().sum())
    #print(np.argwhere(np.isinf(diff)))
    #print(np.argwhere(np.isinf(diff.dropna().drop(22405).drop(22429).drop(22404))))
    
    # Plot PDF (histogram)
    
    # Cut from 01.01.2014
    diff_main = np.zeros((data_all.shape[0]-1,24,133),dtype=np.float64)
    data_scaler = MinMaxScaler()
    diff_drop = data_scaler.fit_transform(diff_drop.values.reshape(-1,1)).reshape(-1)
    diff_main[:,:,0:1] = diff_drop.reshape(-1,24,1)
    diff_main[:,:,1:] = data_all[1:,:,1:]
    np.savez('diff_whole.npz',diff_main)
    plt.hist(diff_drop,bins='sqrt',histtype = 'step',color='r')
    plt.title("PDF of the Difference Series")
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.show()
    plt.plot(diff_drop)
    plt.show()
    plt.boxplot(diff_drop)
    plt.show()

    
    return diff_main, data_all, data_scaler
