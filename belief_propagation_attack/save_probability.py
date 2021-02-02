# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:59:53 2021

@author: kahg8
"""
import sys
import os
import os.path
from test_models import TestModels
from utility import *

import timing

from numpy import savetxt


##### ATTENTION #############
# cm 5 ->8 a refaire (pas♣ de x10 dans l'optimizer sans doute) mc 7 8 p 31 32

def save_probability_list(num_traces):
    model_tester = TestModels(use_extra = False)
    handler = model_tester.real_trace_handler

    for model_file in sorted(listdir(MODEL_FOLDER)):
        out_list = ['cm5','cm6','cm7','cm8','mc7','mc8','p31','p32']
      
        print(model_file)
        var_name = get_variable_name(model_file)
        var = var_name + str(get_variable_number(model_file))
        if not var in out_list:
            continue

        print 'Saving probabilities for : ' + var
        output_list , prob_list , rank_list = handler.get_leakage_rank_list_with_specific_model(MODEL_FOLDER +model_file, traces=num_traces,ASCAD= False,save_proba = True)
        print "> Median Rank: {}".format(np.median(rank_list))
        print "> Median Prob: {}".format(np.median(prob_list))
        if not var_name in listdir(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER + var_name + '/')
        savetxt(OUTPUT_FOLDER + var_name + '/' + var +'.csv', output_list, delimiter=',')
        tf.keras.backend.clear_session()
        

save_probability_list(10000)