# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 08:11:54 2022

@author: martho
"""

from utility import *
import pandas as pd
from tqdm import tqdm
traces = load_trace_data(filepath = TRACEDATA_EXTRA_FILEPATH)
print(traces.shape)
file = 'tracedata_pd_extra.csv'
df = pd.DataFrame(data=traces)
print(df.shape)
print(len(df))
print(len(df[0]))
df.to_csv(file)