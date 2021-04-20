# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:02:03 2021

@author: martho
"""
import os.path
import os
import sys
import h5py
import numpy as np
import argparse
import timing
from time import time
import matplotlib
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from sklearn import preprocessing
from tensorflow.keras import backend as K



from tensorflow.keras.optimizers import RMSprop,Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from numpy import genfromtxt
from utility import *

tf.random.set_seed(7)
np.random.seed(7)

def test_variable_model(variable):
    var_name, var_number, _ = split_variable_name(variable)
    model = load_model(MODEL_FOLDER + variable  + '_test.h5', custom_objects={'tf_rank_loss': tf_rank_loss})
    folder = 'output/s/'
    s_val = {}
    

    real_values = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var_name))[var_number-1]

    validation_data = []
    validation_label = []
    labels = real_values[190000:]

    
    for file in os.listdir(folder):
        if '_rand' in file and not '_training' in file:
            num = int(file.split('_')[0].replace('s',''))
            s_val[num] = genfromtxt(folder + file, delimiter=',')
    for i in range(0,10000):
        temp = []
        for num in range(1,17):
            temp.append(s_val[num][i])
        validation_data.append(temp)



    validation_data = np.array(validation_data)
    validation_label = np.array(validation_label)
    rank_list = []
    prob_list = []
    for i in range(10000):
        
        leakage = model.predict(np.array([validation_data[i]]))[0]

        rank = get_rank_from_prob_dist(leakage, real_values[10000 - i])      
        rank_list.append(rank)
        prob_list.append(probability)
    print('Median rank : ',np.median(rank_list))
    print('Median proba : ',np.median(prob_list))

test_variable_model('s001')

