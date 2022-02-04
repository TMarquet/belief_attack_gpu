# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 08:11:54 2022

@author: martho
"""

from utility import *
import csv
from tqdm import tqdm
traces = load_trace_data()
print(traces.shape)
file = 'tracedata.csv'
with open(file,'w') as f:
    writer = csv.writer(f)
    writer.writerows(traces)
        
    f.close()