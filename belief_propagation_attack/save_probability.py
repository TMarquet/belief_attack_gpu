# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:59:53 2021

@author: kahg8
"""
import sys
import os
import os.path

from utility import *

import timing

from numpy import savetxt
def save_probability_list(num_traces):
    model_tester = TestModels(jitter=JITTER, use_extra=(not RANDOM_KEY) and USE_EXTRA, no_print=not DEBUG, verbose=VERBOSE, histogram=HISTOGRAM)
    handler = model_tester.real_trace_handler
    
    for model_file in sorted(listdir(MODEL_FOLDER + 'adagrad/')):
        var_name = get_variable_name(model_file)
        var = var_name + str(get_variable_number(model_file))
        print 'Saving probabilities for : ' + var
        rank_list, prob_list, predicted_values = handler.get_leakage_rank_list_with_specific_model(model_file, traces=num_traces,ASCAD= True)
        if not get_variable_name(model_file) in listdir(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER + var_name + '/')
        savetxt(OUTPUT_FOLDER + var_name + '/' + var +'.csv', prob_list, delimiter=',')
        

save_probability_list(10000)