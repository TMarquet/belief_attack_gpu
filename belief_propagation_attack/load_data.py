# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 08:11:54 2022

@author: martho
"""

from utility import *
import csv
import tqdm as tqdm
traces = load_trace_data()
print(traces.shape)
file = 'tracedata.csv'
with open(file,'w') as f:
    writer = csv.writer(f)
    for trace in tqdm(traces):
        writer.writerow(trace)
        
    f.close()