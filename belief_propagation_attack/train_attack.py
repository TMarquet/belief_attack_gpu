# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:40:42 2021

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


###########################################################################

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            with self.val_writer.as_default():
                tf.summary.scalar(name,value,step=epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)
############# Loss functions #############










def mlp_new(input_length=700, learning_rate=0.00001, classes=256, loss_function='categorical_crossentropy'):

    if loss_function is None:
        loss_function='rank_loss'
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=input_length, activation='relu'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block1_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Dense(512))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block{}_batchnorm'.format(str(2))))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Dense(1024))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block{}_batchnorm'.format(str(3))))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Dense(512))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block{}_batchnorm'.format(str(4))))
    model.add(tf.keras.layers.Activation('relu'))    
    model.add(Dense(classes, activation='softmax'))
    
    model.add(Dense(classes, activation='softmax', name='predictions'))
    # Save image!
    #plot_model(model, to_file='output/model_plot.png', show_shapes=True, show_layer_names=True)

    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    if loss_function=='rank_loss':
        model.compile(loss=tf_rank_loss, optimizer=optimizer, metrics=['accuracy'])
    elif loss_function=='median_probability_loss':
        model.compile(loss=tf_median_probability_loss, optimizer=optimizer, metrics=['accuracy'])
    else:
        try:
            model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        except ValueError:
            print "!!! Loss Function '{}' not recognised, aborting\n".format(loss_function)
            raise
    return model


### CNN Best model
def cnn_best(input_length=2000, learning_rate=0.00001, filters = 3, classes=256, dense_units=2048,pooling = [0,1,2,3,4],dense_layers = 2,size = [64,128,256,512,512]):
    # From VGG16 design
    input_shape = (input_length, 1)
    model = tf.keras.Sequential(name='cnn')
    
    # Convolution blocks
    
    for i in range(len(size)):  
        if i == 0:
            model.add(Conv1D(size[i], filters, padding='same', name='block{}_conv'.format(i+1),input_shape=input_shape))           
        else:
            model.add(Conv1D(size[i], filters, padding='same', name='block{}_conv'.format(i+1)))          
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        if i in pooling:
            model.add(AveragePooling1D(2, strides=2, name='block{}_pool'.format(i+1)))
    

    model.add(Flatten(name='flatten'))
        
    # Two Dense layers
    
 
    for i in range(0,dense_layers):
        model.add(Dense(dense_units, name='fc{}'.format(i)))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization(name='block_dense{}_batchnorm'.format(i)))
        model.add(tf.keras.layers.Activation('relu'))
       

    model.add(Dense(classes, activation='softmax', name='predictions'))

    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=tf_median_probability_loss, optimizer=optimizer, metrics=['accuracy'])
    return model





#### Training high level function
def train_model(X_profiling, Y_profiling):


    history = model.fit(x=Reshaped_X_profiling, y=reshaped_y, batch_size=batch_size, verbose = progress_bar, epochs=epochs, callbacks=callbacks, validation_data=(Reshaped_validation_data, reshaped_val),use_multiprocessing=True)
    model.save(save_file_name)
    return history

# def train_svm()


def train_variable_model(variable):
    var_name, var_number, _ = split_variable_name(variable)

    folder = 'output/s/'
    s = {}
    real_values = np.load('{}{}.npy'.format(REALVALUES_FOLDER, var_name), allow_pickle=True)[var_number-1,:]
    all_data = []
    all_label = real_values[-10000:]
    print(all_label.shape)    
    for file in os.listdir(folder):
        if '_rand' in file:
            num = int(file.split('_')[0].replace('s',''))
            s[num] = genfromtxt(folder + file, delimiter=',')
    

            



    # ### CNN training
    # if cnn:
    #     # TODO: Test New CNN!
    #     # cnn_best_model = cnn_best(input_length=input_length, learning_rate=learning_rate, classes=classes)
    #     sizes = [[20,40,80]]
    #     pooling = [[2]]
    #     filters = [3]
    #     dense_layers = [3]
    #     dense_units = [4000]            
    #     for size in sizes:
    #         for pool in pooling:
    #             for filter_cnn in filters:
    #                 for layer in dense_layers:
    #                     for unit in dense_units:
    #                         print('Training for {} size, {} pool, {} filter, {} layers, {} units'.format(size,pool,filter_cnn,layer,unit))
    #                         cnn_best_model = cnn_best(input_length=input_length, learning_rate=learning_rate, classes=classes,size=size,dense_layers=layer,dense_units=unit,pooling = pool,filters = filter_cnn)
    #                         cnn_epochs = epochs if epochs is not None else 75
    #                         cnn_batchsize = batch_size
    #                         train_model(X_profiling, Y_profiling, cnn_best_model, store_directory +
    #                                     "{}_cnn{}{}_model1_window{}_size{}_pooling{}_densel{}_denseu{}_filter{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}.h5".format(
    #                                         'all_s_proba', hammingweight_flag, hammingdistance_flag, input_length, sizes.index(size),pooling.index(pool),layer,unit,filter_cnn, cnn_batchsize, learning_rate, sd, training_traces, augment_method, jitter),
    #                                     epochs=cnn_epochs, batch_size=cnn_batchsize, validation_data=(X_attack, Y_attack),
    #                                     progress_bar=progress_bar, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)

    # ### MLP training
    # elif mlp:
    #     if multilabel:
    #         mlp_best_model = mlp_weighted_bit(input_length=input_length, layer_nb=mlp_layers, learning_rate=learning_rate, classes=classes, loss_function=loss_function)
    #     else:
    #         mlp_best_model = mlp_new(input_length=input_length, learning_rate=learning_rate, classes=classes, loss_function=loss_function)
    #     mlp_epochs = epochs if epochs is not None else 200
    #     mlp_batchsize = batch_size
    #     train_model(X_profiling, Y_profiling, mlp_best_model, store_directory +
    #                 "{}_mlp{}{}{}{}_nodes{}_window{}_epochs{}_batchsize{}_lr{}_sd{}_traces{}_aug{}_jitter{}_{}.h5".format(
    #                     variable, mlp_layers, '_multilabel' if multilabel else '', hammingweight_flag, hammingdistance_flag, mlp_nodes, input_length, mlp_epochs, mlp_batchsize, learning_rate, sd,
    #                     training_traces, augment_method, jitter, 'defaultloss' if loss_function is None else loss_function.replace('_','')), epochs=mlp_epochs, batch_size=mlp_batchsize,
    #                 validation_data=(X_attack, Y_attack), progress_bar=progress_bar, multilabel=multilabel, hammingweight=hammingweight, hamming_distance_encoding=hamming_distance_encoding)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--MLP', action="store_true", dest="USE_MLP", help='Trains Multi Layer Perceptron',
                        default=False)
    parser.add_argument('--CNN', action="store_true", dest="USE_CNN",
                        help='Trains Convolutional Neural Network', default=False)


    # Target node here
    args            = parser.parse_args()
    USE_MLP         = args.USE_MLP
    USE_CNN         = args.USE_CNN
 



    variable_list =['s001','s002','s003','s004','s005','s006','s007','s008','s009','s010','s011','s012','s013','s014','s015','s016']


    X = np.array([])
    X_l =  np.array([])
    V = np.array([])
    V_l =  np.array([])

    train_variable_model('s001')         



    print "$ Done!"
