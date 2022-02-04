# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 08:11:54 2022

@author: martho
"""

from utility import *
import csv
traces = load_trace_data()
print(traces.shape)
file = 'tracedata.csv'
with open(file,'w',encoding='UTF8') as f:
    write = csv.write(file)
    for trace in traces:
        writer.writerow(trace)