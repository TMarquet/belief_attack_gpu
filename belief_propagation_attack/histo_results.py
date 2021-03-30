# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:55:33 2020

@author: kahg8
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


var_to_plot = ['s','k','t','p']
all_v = ['s','k','t','h','p','cm','mc','xt','p2']
#var_to_plot = all_v

def dist(x1,x2):
    return np.sqrt(pow(x1[0]-x2[0],2)+pow(x1[1]-x2[1],2))/256

def average_nn(l):
    sum_nn = 0
    for elem1 in l:
        min_v = 9999999999
        for elem2 in l:
            d = 9999999999
            if elem1 != elem2 :
                d = dist(elem1,elem2)
            if min_v > d :
                min_v = d
        sum_nn += min_v
    return sum_nn/len(l)

def evaluate_consitency(l):
    max_l = max(l)
    min_l = min(l)
    mean_l = np.mean(l)
    median_l = np.median(l)
    return np.sqrt(pow(mean_l-max_l,2)+pow(mean_l-min_l,2))/256


########### MY VALUES ####################


s_median_rank = [47,26,58,75,42,78,86,88,41,66,73,67,39,78,104,93]
k_median_rank = [120,106,114,100,118,102,127,110,125,63,127,104,80,57,100,69,73,72,88,80,83,85,81,87,86,83,82,83,85,84,87,95]
p_median_rank = [122,66,127,121,124,65,125,118,121,62,121,113,107,60,111,68]
p_2_round_rank = []
t_median_rank = [68,77,77,76,68,68,67,68,62,65,66,65,64,63,65,61,69,69,104,122,120,116,120,120,118,113,119,119,118,109,114,113]
h_median_rank = []
xt_median_rank = []
xk_median_rank = []
cm_median_rank = []
mc_median_rank = []
sk_median_rank = []

median_rank = {}

median_rank['s'] = s_median_rank
median_rank['k'] = k_median_rank
median_rank['p'] = p_median_rank
median_rank['t'] = t_median_rank
median_rank['h'] = h_median_rank
median_rank['cm'] = cm_median_rank
median_rank['mc'] = mc_median_rank
median_rank['sk'] = sk_median_rank
median_rank['xk'] = xk_median_rank
median_rank['xt'] = xt_median_rank

s_median_proba = [0.60,1.03,0.50,0.49,0.70,0.48,0.43,0.41,0.74,0.54,0.52,0.52,0.80,0.40,0.29,0.41]
k_median_proba = [0.39,0.40,0.34,0.40,0.38,0.43,0.35,0.40,0.36,0.73,0.34,0.40,0.34,0.77,0.41,0.45,0.43,0.52,0.41,0.46,0.46,0.43,0.44,0.39,0.43,0.42,0.43,0.42,0.43,0.31,0.42,0.38]
p_median_proba = [0.39,0.70,0.39,0.39,0.39,0.70,0.39,0.39,0.39,0.73,0.39,0.39,0.40,0.74,0.40,0.51]
p_2_round_proba = []
t_median_proba = [0.52,0.48,0.48,0.48,0.52,0.51,0.52,0.51,0.55,0.53,0.54,0.54,0.55,0.70,0.55,0.57,0.50,0.48,0.41,0.35,0.37,0.02,0.02,0.007,0.009,0.02,0.03,0.03,0.006,0.16,0.15,0.11]
print(len(k_median_proba))
h_median_proba = []
xt_median_proba = []
xk_median_proba = []
cm_median_proba = []
mc_median_proba = []
sk_median_proba = []

median_proba = {}

median_proba['s'] = s_median_proba  
median_proba['k'] = k_median_proba
median_proba['p'] = p_median_proba

median_proba['t'] = t_median_proba
median_proba['h'] = h_median_proba
median_proba['cm'] = cm_median_proba
median_proba['mc'] = mc_median_proba
median_proba['sk'] = sk_median_proba
median_proba['xk'] = xk_median_proba
median_proba['xt'] = xt_median_proba

added = False
if 'p2' in var_to_plot :
    
    if 'p' in var_to_plot :
        
        median_rank["p"] +=  p_2_round_rank
        median_proba['p'] += p_2_round_proba
    else :
        median_rank["p"] =  p_2_round_rank
        median_proba['p'] = p_2_round_proba
        var_to_plot.append('p')
        added = True       
    var_to_plot.remove('p2')




############### MLP Values ####################

s_MLP_median_rank = [57,39,78,73,49,82,78,88,47,71,82,71,50,65,81,80]
k_MLP_median_rank = [104,104,98,95,116,87,118,97,110,62,118,94,77,63,111,85]
p_MLP_median_rank = [119,70,122,119,114,66,120,114,115,63,113,111,115,65,104,80,88,68,63,76,121,112,124,100,96,76,112,105,121,98,116,86]
t_MLP_median_rank = [70,78,84,77,79,88,69,71,62,63,67,65,63,65,67,58]
h_MLP_median_rank = [81,107,113,72,94,94,77,99,79,72,118,74,95,97,102,83]
xt_MLP_median_rank = [87,75,71,76,102,91,62,101,91,70,88,87,99,73,74,88]
xk_MLP_median_rank = [56]
cm_MLP_median_rank = [108,118,114,102,106,123,105,107,106,111,104,101,114,113,111,120]
mc_MLP_median_rank = [80,97,76,74,77,77,63,74,97,70,72,93,73,83,70,79]
sk_MLP_median_rank = [65]

MLP_median_rank = {}

MLP_median_rank['s'] = s_MLP_median_rank
MLP_median_rank['k'] = k_MLP_median_rank
MLP_median_rank['p'] = p_MLP_median_rank
MLP_median_rank['t'] = t_MLP_median_rank
MLP_median_rank['h'] = h_MLP_median_rank
MLP_median_rank['cm'] = cm_MLP_median_rank
MLP_median_rank['mc'] = mc_MLP_median_rank
MLP_median_rank['sk'] = sk_MLP_median_rank
MLP_median_rank['xk'] = xk_MLP_median_rank
MLP_median_rank['xt'] = xt_MLP_median_rank

s_MLP_median_proba = [0.61,0.79,0.37,0.58,0.70,0.50,0.52,0.46,0.72,0.58,0.53,0.57,0.71,0.77,0.56,0.53]
k_MLP_median_proba = [0.41,0.42,0.44,0.45,0.40,0.48,0.39,0.42,0.41,0.75,0.39,0.42,0.51,0.74,0.40,0.46]
p_MLP_median_proba = [0.39,0.68,0.38,0.38,0.40,0.70,0.40,0.40,0.40,0.71,0.39,0.40,0.40,0.67,0.41,0.48,0.53,0.59,0.56,0.55,0.52,0.33,0.17,0.47,0.51,0.57,0.40,0.44,0.24,0.47,0.36,0.51]
t_MLP_median_proba = [0.53,0.52,0.49,0.52,0.52,0.49,0.54,0.53,0.57,0.55,0.54,0.53,0.54,0.53,0.52,0.56]
h_MLP_median_proba = [0.53,0.43,0.31,0.57,0.45,0.50,0.53,0.44,0.59,0.54,0.41,0.62,0.45,0.45,0.43,0.53]
xt_MLP_median_proba = [0.50,0.52,0.55,0.51,0.44,0.48,0.61,0.44,0.47,0.55,0.40,0.51,0.43,0.53,0.55,0.50]
xk_MLP_median_proba = [0.38]
cm_MLP_median_proba = [0.42,0.4,0.4,0.43,0.36,0.42,0.42,0.42,0.41,0.42,0.43,0.41,0.4,0.41,0.4,0.41,0.4]
mc_MLP_median_proba = [0.52,0.45,0.52,0.51,0.54,0.52,0.61,0.53,0.45,0.55,0.54,0.47,0.55,0.50,0.56,0.52]
sk_MLP_median_proba = [0.28]

MLP_median_proba = {}

MLP_median_proba['s'] = s_MLP_median_proba
MLP_median_proba['k'] = k_MLP_median_proba
MLP_median_proba['p'] = p_MLP_median_proba
MLP_median_proba['t'] = t_MLP_median_proba
MLP_median_proba['h'] = h_MLP_median_proba
MLP_median_proba['cm'] = cm_MLP_median_proba
MLP_median_proba['mc'] = mc_MLP_median_proba
MLP_median_proba['sk'] = sk_MLP_median_proba
MLP_median_proba['xk'] = xk_MLP_median_proba
MLP_median_proba['xt'] = xt_MLP_median_proba


###############################################


dict_val_rank = {}
dict_MLP_val_rank = {}
dict_val_proba = {}
dict_MLP_val_proba = {}

print('Studied variable :',var_to_plot)

for elem in var_to_plot :
    i = 1
    for e in median_proba[elem] :
        dict_val_proba[elem+str(i)] = e *pow(10,-2)
        dict_MLP_val_proba[elem+str(i)] = MLP_median_proba[elem][i-1] *pow(10,-2)
        i+=1
    i = 1 
    for e in median_rank[elem] :
        dict_val_rank[elem+str(i)] = e
        dict_MLP_val_rank[elem+str(i)] = MLP_median_rank[elem][i-1] 
        i+=1


labels = dict_val_proba.keys()
CNN_proba = dict_val_proba.values()
MLP_proba = dict_MLP_val_proba.values()
CNN_rank = dict_val_rank.values()
MLP_rank = dict_MLP_val_rank.values()
X_CNN =[]
X_MLP = []


a = np.array(list(CNN_rank))
b = np.array(list(CNN_proba))
c = np.array(list(MLP_rank))
d = np.array(list(MLP_proba))


for x , y in zip(CNN_proba,CNN_rank) :
    X_CNN.append([x,y])
    
for x , y in zip(MLP_proba,MLP_rank) :
    X_MLP.append([x,y])

print('CNN median rank mean : ',np.mean(a))
print('CNN median proba mean : ',np.mean(b))
print('\n')
print('MLP median rank mean : ',np.mean(c))
print('MLP median proba mean : ',np.mean(d))
print('\n')
print('CNN median rank median : ',np.median(a))
print('CNN median proba median : ',np.median(b))
print('\n')
print('MLP median rank median : ',np.median(c))
print('MLP median proba median : ',np.median(d))
print('\n')
print('CNN median rank min: ',np.min(a))
print('CNN median proba min : ',np.min(b))
print('\n')
print('MLP median rank min : ',np.min(c))
print('MLP median proba min : ',np.min(d))
print('\n')
print('CNN median rank max : ',np.max(a))
print('CNN median proba max : ',np.max(b))
print('\n')
print('MLP median rank max : ',np.max(c))
print('MLP median proba max : ',np.max(d))

plt.figure(1)
plt.scatter(CNN_rank, CNN_proba,c='blue', s=0.5)
plt.scatter(MLP_rank, MLP_proba,c='red', s=0.5)
ax_sca = plt.gca()
ax_sca.set_xlim(0,150)
ax_sca.set_ylim(0,0.25)



if len(var_to_plot) > 1 :

    fig, ax = plt.subplots(len(var_to_plot),2)
    plt.subplots_adjust(hspace=0.5)
    #fig, ax = plt.subplots(3,1)
    
    for plot in range(0,len(var_to_plot)):
        labels = []
        start = 0
        n = len(median_rank[var_to_plot[plot]])
        if added :
            start = 16


        for i in range(0,n):
            labels.append(var_to_plot[plot] + str(start + i+1))    
        x = np.arange(n)  # the label locations
        width = 0.35  # the width of the bars
        rects1 = ax[plot][0].bar(x - width/2, median_rank[var_to_plot[plot]], width, label='CNN')
        rects2 = ax[plot][0].bar(x + width/2, MLP_median_rank[var_to_plot[plot]][:n], width, label='MLP')
        rects1 = ax[plot][1].bar(x - width/2, median_proba[var_to_plot[plot]], width, label='CNN')
        rects2 = ax[plot][1].bar(x + width/2, MLP_median_proba[var_to_plot[plot]][:n], width, label='MLP')
        CNN_rank_mean = np.mean(median_rank[var_to_plot[plot]])
        MLP_rank_mean = np.mean(MLP_median_rank[var_to_plot[plot]])
        line_CNN = ax[plot][0].plot([-width,n-1+width],[CNN_rank_mean,CNN_rank_mean],'r--',label='CNN mean')
        line_MLP = ax[plot][0].plot([-width,n-1+width],[MLP_rank_mean,MLP_rank_mean],'r:',label='MLP mean')
        
        CNN_proba_mean = np.mean(median_proba[var_to_plot[plot]])
        MLP_proba_mean = np.mean(MLP_median_proba[var_to_plot[plot]])
        line_CNN = ax[plot][1].plot([-width,n-1+width],[CNN_proba_mean,CNN_proba_mean],'r--',label='CNN mean')
        line_MLP = ax[plot][1].plot([-width,n-1+width],[MLP_proba_mean,MLP_proba_mean],'r:',label='MLP mean')    
    # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[plot][0].set_ylabel('Rank')
        ax[plot][0].set_ylim(0,150)
        ax[plot][0].set_title('Rank comparison between CNN and MLP for var '+ var_to_plot[plot])
        ax[plot][0].set_xticks(x)
        ax[plot][0].set_xticklabels(labels)
    
        ax[plot][1].set_ylabel('Probability in % ')
    
        ax[plot][1].set_title('Median probability comparison between CNN and MLP for var '+ var_to_plot[plot])
        ax[plot][1].set_xticks(x)
        ax[plot][1].set_xticklabels(labels)
    

    handles, labels = ax[0][0].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='upper left',fancybox=True, framealpha=1)
else :
    
    fig, ax = plt.subplots(2,1)
    plot = 0
    labels = []
    start = 0
    n = len(median_rank[var_to_plot[plot]])
    if added :
        start = 16
    for i in range(0,n):
        labels.append(var_to_plot[plot] + str(start+i+1))   
    x = np.arange(n)  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax[0].bar(x - width/2, median_rank[var_to_plot[plot]], width, label='CNN')
    rects2 = ax[0].bar(x + width/2, MLP_median_rank[var_to_plot[plot]][:n], width, label='MLP')
    rects1 = ax[1].bar(x - width/2, median_proba[var_to_plot[plot]], width, label='CNN')
    rects2 = ax[1].bar(x + width/2, MLP_median_proba[var_to_plot[plot]][:n], width, label='MLP')
    CNN_rank_mean = np.mean(median_rank[var_to_plot[plot]])
    MLP_rank_mean = np.mean(MLP_median_rank[var_to_plot[plot]])
    line_CNN = ax[0].plot([-width,n-1+width],[CNN_rank_mean,CNN_rank_mean],'r--',label='CNN mean')
    line_MLP = ax[0].plot([-width,n-1+width],[MLP_rank_mean,MLP_rank_mean],'r:',label='MLP mean')
    
    CNN_proba_mean = np.mean(median_proba[var_to_plot[plot]])
    MLP_proba_mean = np.mean(MLP_median_proba[var_to_plot[plot]])
    line_CNN = ax[1].plot([-width,n-1+width],[CNN_proba_mean,CNN_proba_mean],'r--',label='CNN mean')
    line_MLP = ax[1].plot([-width,n-1+width],[MLP_proba_mean,MLP_proba_mean],'r:',label='MLP mean')    
# Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Rank')
    ax[0].set_ylim(0,150)
    ax[0].set_title('Rank comparison between CNN and MLP for var '+ var_to_plot[plot])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)

    ax[1].set_ylabel('Probability in % ')

    ax[1].set_title('Median probability comparison between CNN and MLP for var '+ var_to_plot[plot])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left',fancybox=True, framealpha=1)
    plt.tight_layout()
    
    
plt.show()