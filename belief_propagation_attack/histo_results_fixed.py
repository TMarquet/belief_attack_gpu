# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:55:33 2020

@author: kahg8
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import smallest_circle as sc
from scipy.spatial import ConvexHull, convex_hull_plot_2d
var_to_plot = ['s']
all_v = ['s','k','t','h','p','cm','mc','xt']
#var_to_plot = all_v

########### MY VALUES ####################


s_median_rank = []
k_median_rank = []
p_median_rank = []
t_median_rank = []
h_median_rank = []
xt_median_rank = []
cm_median_rank = []
mc_median_rank = [] 


median_rank = {}

median_rank['s'] = s_median_rank
median_rank['k'] = k_median_rank
median_rank['p'] = p_median_rank
median_rank['t'] = t_median_rank
median_rank['h'] = h_median_rank
median_rank['cm'] = cm_median_rank
median_rank['mc'] = mc_median_rank
median_rank['xt'] = xt_median_rank

s_median_proba = []
k_median_proba = []
p_median_proba = []
t_median_proba = []
h_median_proba = []
xt_median_proba = []
cm_median_proba = []

mc_median_proba = []


median_proba = {}

median_proba['s'] = s_median_proba  
median_proba['k'] = k_median_proba
median_proba['p'] = p_median_proba

median_proba['t'] = t_median_proba
median_proba['h'] = h_median_proba
median_proba['cm'] = cm_median_proba
median_proba['mc'] = mc_median_proba
median_proba['xt'] = xt_median_proba



############### CNN Combined Values ####################

k_COMBINE_median_rank =[]
s_COMBINE_median_rank = []
p_COMBINE_median_rank = []
t_COMBINE_median_rank =[]
h_COMBINE_median_rank = []
xt_COMBINE_median_rank =[] 
cm_COMBINE_median_rank = []
mc_COMBINE_median_rank = []


COMBINE_median_rank = {}

COMBINE_median_rank['s'] = s_COMBINE_median_rank
COMBINE_median_rank['k'] = k_COMBINE_median_rank
COMBINE_median_rank['p'] = p_COMBINE_median_rank
COMBINE_median_rank['t'] = t_COMBINE_median_rank
COMBINE_median_rank['h'] = h_COMBINE_median_rank
COMBINE_median_rank['cm'] = cm_COMBINE_median_rank
COMBINE_median_rank['mc'] = mc_COMBINE_median_rank
COMBINE_median_rank['xt'] = xt_COMBINE_median_rank

k_COMBINE_median_proba = []
s_COMBINE_median_proba = []
p_COMBINE_median_proba = []
t_COMBINE_median_proba =[]
h_COMBINE_median_proba = []
xt_COMBINE_median_proba =[]

cm_COMBINE_median_proba = []

mc_COMBINE_median_proba = []


COMBINE_median_proba = {}

COMBINE_median_proba['s'] = s_COMBINE_median_proba
COMBINE_median_proba['k'] = k_COMBINE_median_proba
COMBINE_median_proba['p'] = p_COMBINE_median_proba
COMBINE_median_proba['t'] = t_COMBINE_median_proba
COMBINE_median_proba['h'] = h_COMBINE_median_proba
COMBINE_median_proba['cm'] = cm_COMBINE_median_proba
COMBINE_median_proba['mc'] = mc_COMBINE_median_proba
COMBINE_median_proba['xt'] = xt_COMBINE_median_proba


###############################################



############### MLP Values ####################

s_MLP_median_rank = []
k_MLP_median_rank = []
p_MLP_median_rank = []
t_MLP_median_rank =[]
h_MLP_median_rank = []
xt_MLP_median_rank = []
cm_MLP_median_rank = []
mc_MLP_median_rank = []

MLP_median_rank = {}

MLP_median_rank['s'] = s_MLP_median_rank
MLP_median_rank['k'] = k_MLP_median_rank
MLP_median_rank['p'] = p_MLP_median_rank
MLP_median_rank['t'] = t_MLP_median_rank
MLP_median_rank['h'] = h_MLP_median_rank
MLP_median_rank['cm'] = cm_MLP_median_rank
MLP_median_rank['mc'] = mc_MLP_median_rank
MLP_median_rank['xt'] = xt_MLP_median_rank

s_MLP_median_proba =[]
k_MLP_median_proba = []
p_MLP_median_proba = []
t_MLP_median_proba = []
h_MLP_median_proba = []
xt_MLP_median_proba =[]
cm_MLP_median_proba = []
mc_MLP_median_proba =[] 

MLP_median_proba = {}

MLP_median_proba['s'] = s_MLP_median_proba
MLP_median_proba['k'] = k_MLP_median_proba
MLP_median_proba['p'] = p_MLP_median_proba
MLP_median_proba['t'] = t_MLP_median_proba
MLP_median_proba['h'] = h_MLP_median_proba
MLP_median_proba['cm'] = cm_MLP_median_proba
MLP_median_proba['mc'] = mc_MLP_median_proba
MLP_median_proba['xt'] = xt_MLP_median_proba


###############################################


dict_val_rank = {}
dict_MLP_val_rank = {}
dict_val_proba = {}
dict_MLP_val_proba = {}
dict_combine_val_rank = {}
dict_combine_val_proba = {}
print('Studied variable :',var_to_plot)

for elem in var_to_plot :
    i = 1
    for e in median_proba[elem] :
        dict_val_proba[elem+str(i)] = e *pow(10,-2) /0.0039
        dict_MLP_val_proba[elem+str(i)] = MLP_median_proba[elem][i-1] *pow(10,-2)/0.0039
        dict_combine_val_proba[elem+str(i)] = COMBINE_median_proba[elem][i-1] *pow(10,-2)/0.0039
        i+=1
    i = 1 
    for e in median_rank[elem] :
        dict_val_rank[elem+str(i)] = e / 128
        dict_MLP_val_rank[elem+str(i)] = MLP_median_rank[elem][i-1] / 128
        dict_combine_val_rank[elem+str(i)] = COMBINE_median_rank[elem][i-1] / 128
        i+=1


labels = dict_val_proba.keys()
CNN_proba = dict_val_proba.values()
MLP_proba = dict_MLP_val_proba.values()
CNN_Combine_proba = dict_combine_val_proba.values()
CNN_rank = dict_val_rank.values()
MLP_rank = dict_MLP_val_rank.values()
CNN_combine_rank = dict_combine_val_rank.values()
X_CNN =[]
X_MLP = []


a = np.array(list(CNN_rank))
b = np.array(list(CNN_proba))
c = np.array(list(MLP_rank))
d = np.array(list(MLP_proba))
e = np.array(list(CNN_combine_rank))
f = np.array(list(CNN_Combine_proba))


print('CNN median rank mean : ',np.mean(a))
print('CNN median proba mean : ',np.mean(b))
print('\n')
print('CNN combine median rank mean : ',np.mean(e))
print('CNN combine  median proba mean : ',np.mean(f))
print('\n')
print('MLP median rank mean : ',np.mean(c))
print('MLP median proba mean : ',np.mean(d))
print('\n')
print('CNN median rank median : ',np.median(a))
print('CNN median proba median : ',np.median(b))
print('\n')
print('CNN combine median rank median : ',np.median(e))
print('CNN combine median proba median : ',np.median(f))
print('\n')
print('MLP median rank median : ',np.median(c))
print('MLP median proba median : ',np.median(d))
print('\n')
print('CNN median rank min: ',np.min(a))
print('CNN median proba min : ',np.min(b))
print('\n')
print('CNN combine median rank min: ',np.min(e))
print('CNN median proba min : ',np.min(f))
print('\n')
print('MLP median rank min : ',np.min(c))
print('MLP median proba min : ',np.min(d))
print('\n')
print('CNN median rank max : ',np.max(a))
print('CNN median proba max : ',np.max(b))
print('\n')
print('CNN combine median rank max : ',np.max(e))
print('CNN combine median proba max : ',np.max(f))
print('\n')
print('MLP median rank max : ',np.max(c))
print('MLP median proba max : ',np.max(d))

fig,ax =  plt.subplots()
ax.scatter(CNN_rank, CNN_proba,c='blue', s=10)
points_cnn = np.array(list(zip(CNN_rank,CNN_proba)))
hull_cnn = ConvexHull(points_cnn)
for simplex in hull_cnn.simplices:
    ax.plot(points_cnn[simplex, 0], points_cnn[simplex, 1], 'b-')
ax.scatter(CNN_combine_rank, CNN_Combine_proba,c='green', s=10)
points_cnn_combine = np.array(list(zip(CNN_combine_rank,CNN_Combine_proba)))
hull_cnn_combine = ConvexHull(points_cnn_combine)
for simplex in hull_cnn_combine.simplices:
    ax.plot(points_cnn_combine[simplex, 0], points_cnn_combine[simplex, 1], 'g-')
ax.scatter(MLP_rank, MLP_proba,c='red', s=10)

points_mlp = np.array(list(zip(MLP_rank,MLP_proba)))
hull_mlp = ConvexHull(points_mlp)
for simplex in hull_mlp.simplices:
    ax.plot(points_mlp[simplex, 0], points_mlp[simplex, 1], 'r-')
print('CNN consistency : ',hull_cnn.area)
print('Combined CNN consistency : ',hull_cnn_combine.area)
print('MLP consistency : ',hull_mlp.area)

ax_sca = plt.gca()
ax_sca.set_xlim(0,1)
ax_sca.set_ylim(0,max([np.max(points_cnn),np.max(points_cnn_combine),np.max(points_mlp)])*1.1)



if len(var_to_plot) > 1 :

    fig, ax = plt.subplots(len(var_to_plot),2)
    plt.subplots_adjust(hspace=0.5)
    #fig, ax = plt.subplots(3,1)
    
    for plot in range(0,len(var_to_plot)):
        labels = []
        start = 0
        n = len(median_rank[var_to_plot[plot]])



        for i in range(0,n):
            labels.append(var_to_plot[plot] + str(start + i+1))    
        x = np.arange(n)  # the label locations
        width = 0.2  # the width of the bars
        rects1 = ax[plot][0].bar(x - width/2, median_rank[var_to_plot[plot]], width, label='CNN')
        rects2 = ax[plot][0].bar(x , MLP_median_rank[var_to_plot[plot]][:n], width, label='MLP')
        rects3 = ax[plot][0].bar(x + width/2, COMBINE_median_rank[var_to_plot[plot]][:n], width, label='CNN combined')
        rects1 = ax[plot][1].bar(x - width/2, median_proba[var_to_plot[plot]], width, label='CNN')
        rects2 = ax[plot][1].bar(x , MLP_median_proba[var_to_plot[plot]][:n], width, label='MLP')
        rects3 = ax[plot][1].bar(x + width/2, COMBINE_median_proba[var_to_plot[plot]][:n], width, label='CNN combined')
        CNN_rank_mean = np.mean(median_rank[var_to_plot[plot]])
        MLP_rank_mean = np.mean(MLP_median_rank[var_to_plot[plot]])
        CNN_combine_rank_mean = np.mean(COMBINE_median_rank[var_to_plot[plot]])
        line_CNN = ax[plot][0].plot([-width,n-1+width],[CNN_rank_mean,CNN_rank_mean],'b--',label='CNN mean')
        line_MLP = ax[plot][0].plot([-width,n-1+width],[MLP_rank_mean,MLP_rank_mean],'r:',label='MLP mean')
        line_COMBINE = ax[plot][0].plot([-width,n-1+width],[CNN_combine_rank_mean,CNN_combine_rank_mean],'k-.',label='CNN combine mean')
        CNN_proba_mean = np.mean(median_proba[var_to_plot[plot]])
        MLP_proba_mean = np.mean(MLP_median_proba[var_to_plot[plot]])
        COMBINE_proba_mean = np.mean(COMBINE_median_proba[var_to_plot[plot]])
        line_CNN = ax[plot][1].plot([-width,n-1+width],[CNN_proba_mean,CNN_proba_mean],'b--',label='CNN mean')
        line_MLP = ax[plot][1].plot([-width,n-1+width],[MLP_proba_mean,MLP_proba_mean],'r:',label='MLP mean')    
        line_COMBINE = ax[plot][1].plot([-width,n-1+width],[COMBINE_proba_mean,COMBINE_proba_mean],'k-.',label='CNN combine mean') 
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

    for i in range(0,n):
        labels.append(var_to_plot[plot] + str(start+i+1))   
    x = np.arange(n)  # the label locations
    width = 0.2  # the width of the bars
    rects1 = ax[0].bar(x - width, median_rank[var_to_plot[plot]], width, label='CNN')
    rects2 = ax[0].bar(x , MLP_median_rank[var_to_plot[plot]][:n], width, label='MLP')
    rects3 = ax[0].bar(x + width, COMBINE_median_rank[var_to_plot[plot]][:n], width, label='CNN combined')
    rects1 = ax[1].bar(x - width, median_proba[var_to_plot[plot]], width, label='CNN')
    rects2 = ax[1].bar(x , MLP_median_proba[var_to_plot[plot]][:n], width, label='MLP')
    rects3 = ax[1].bar(x + width, COMBINE_median_proba[var_to_plot[plot]][:n], width, label='CNN combined')
    CNN_rank_mean = np.mean(median_rank[var_to_plot[plot]])
    MLP_rank_mean = np.mean(MLP_median_rank[var_to_plot[plot]])
    CNN_combine_rank_mean = np.mean(COMBINE_median_rank[var_to_plot[plot]])
    line_CNN = ax[0].plot([-width,n-1+width],[CNN_rank_mean,CNN_rank_mean],'b--',label='CNN mean')
    line_MLP = ax[0].plot([-width,n-1+width],[MLP_rank_mean,MLP_rank_mean],'r:',label='MLP mean')
    line_COMBINE = ax[0].plot([-width,n-1+width],[CNN_combine_rank_mean,CNN_combine_rank_mean],'k-.',label='CNN combine mean')
    CNN_proba_mean = np.mean(median_proba[var_to_plot[plot]])
    MLP_proba_mean = np.mean(MLP_median_proba[var_to_plot[plot]])
    COMBINE_proba_mean = np.mean(COMBINE_median_proba[var_to_plot[plot]])
    line_CNN = ax[1].plot([-width,n-1+width],[CNN_proba_mean,CNN_proba_mean],'b--',label='CNN mean')
    line_MLP = ax[1].plot([-width,n-1+width],[MLP_proba_mean,MLP_proba_mean],'r:',label='MLP mean')    
    line_COMBINE = ax[1].plot([-width,n-1+width],[COMBINE_proba_mean,COMBINE_proba_mean],'k-.',label='CNN combine mean') 
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