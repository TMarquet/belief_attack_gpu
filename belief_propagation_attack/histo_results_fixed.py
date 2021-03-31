# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:04:19 2021

@author: martho
"""

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
all_v = ['s','k','t','h','p','cm','mc','xt']
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


s_median_rank = []
k_median_rank = []
p_median_rank = []
p_2_round_rank = []
t_median_rank = []
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

s_median_proba = []
k_median_proba =[]
p_median_proba = []
p_2_round_proba = []
t_median_proba = []

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

s_MLP_median_rank =[]
k_MLP_median_rank =[]
p_MLP_median_rank = []
t_MLP_median_rank =[]
h_MLP_median_rank =[]
xt_MLP_median_rank =[]
xk_MLP_median_rank =[]
cm_MLP_median_rank =[]
mc_MLP_median_rank = []
sk_MLP_median_rank = []

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

s_MLP_median_proba = []
k_MLP_median_proba = []
p_MLP_median_proba = []
t_MLP_median_proba = []
h_MLP_median_proba = []
xt_MLP_median_proba = []
xk_MLP_median_proba =[]
cm_MLP_median_proba = []
mc_MLP_median_proba = []
sk_MLP_median_proba = []

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