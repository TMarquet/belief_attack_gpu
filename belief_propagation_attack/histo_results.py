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
var_to_plot = ['xt']
all_v = ['s','k','t','h','p','cm','mc','xt']
#var_to_plot = all_v


########### MY VALUES ####################


s_median_rank = [47,26,58,75,42,78,86,88,41,66,73,67,39,78,104,93,52,25,51,76,45,80,77,84,39,70,74,68,40,81,100,96]
k_median_rank = [120,106,114,100,118,102,127,110,125,63,127,104,80,57,100,69,73,72,88,80,83,85,81,87,86,83,82,83,85,84,87,95]
p_median_rank = [122,66,127,121,124,65,125,118,121,62,121,113,107,60,111,68,80,124,71,84,82,84,80,119,83,82,78,122,94,92,81,120]
t_median_rank = [68,77,77,76,68,68,67,68,62,65,66,65,64,63,65,61,69,69,104,122,120,116,120,120,118,113,119,119,118,109,114,113]
h_median_rank = [88,109,97,87,106,94,103,110,104,85,117,63]
xt_median_rank = [93.0, 74.0, 68.0, 70.0,94.0, 73.0, 80.0, 69.0,100.5, 68.0, 83.0, 73.0,89.0, 77.0, 85.0, 72.0]
cm_median_rank = [107,124,112,106,106.0, 116.0, 103.0, 112.0,107.0, 118.0, 104.0, 112.0,109.0, 119.0, 104.0, 117.0]
mc_median_rank = [88.0, 73.0, 66.0, 73.0,91.0, 73.0, 78.0, 68.0,104.0, 65.0, 83.0, 72.0,81.0, 75.0, 83.0, 73.0]


median_rank = {}

median_rank['s'] = s_median_rank
median_rank['k'] = k_median_rank
median_rank['p'] = p_median_rank
median_rank['t'] = t_median_rank
median_rank['h'] = h_median_rank
median_rank['cm'] = cm_median_rank
median_rank['mc'] = mc_median_rank
median_rank['xt'] = xt_median_rank

s_median_proba = [0.60,1.03,0.50,0.49,0.70,0.48,0.43,0.41,0.74,0.54,0.52,0.52,0.80,0.40,0.29,0.41,0.56,1.07,0.61,0.50,0.67,0.47,0.48,0.46,0.77,0.53,0.51,0.50,0.77,0.38,0.38,0.39]
k_median_proba = [0.39,0.40,0.34,0.40,0.38,0.43,0.35,0.40,0.36,0.73,0.34,0.40,0.34,0.77,0.41,0.45,0.43,0.52,0.41,0.46,0.46,0.43,0.44,0.39,0.43,0.42,0.43,0.42,0.43,0.31,0.42,0.38]
p_median_proba = [0.39,0.70,0.39,0.39,0.39,0.70,0.39,0.39,0.39,0.73,0.39,0.39,0.40,0.74,0.40,0.51,0.45,0.02,0.51,0.46,0.44,0.36,0.45,0.12,0.42,0.34,0.46,0.09,0.34,0.31,0.45,0.2]
t_median_proba = [0.52,0.48,0.48,0.48,0.52,0.51,0.52,0.51,0.55,0.53,0.54,0.54,0.55,0.70,0.55,0.57,0.50,0.48,0.41,0.35,0.37,0.02,0.02,0.007,0.009,0.02,0.03,0.03,0.006,0.16,0.15,0.11]
h_median_proba = [0.19,0.38,0.23,0.19,0.39,0.24,0.12,0.38,0.17,0.28,0.37,0.51]
xt_median_proba = [0.1724143, 0.38085456, 0.4286984, 0.44241045,0.12759984, 0.4049649, 0.23264163, 0.4594505,0.14931167, 0.4220913, 0.26216935, 0.4435829,0.22457005, 0.37966073, 0.2336064, 0.4740294]
cm_median_proba = [0.4,0.39,0.4,0.4,0.40094256, 0.38626224, 0.40714145, 0.38982127,0.4011262, 0.39268914, 0.40722257, 0.3957341,0.401037, 0.37970073, 0.40965155, 0.3940438]

mc_median_proba = [0.19157205, 0.389596, 0.45824605, 0.4274725,0.15769254, 0.41880417, 0.2618316, 0.4692219,0.11663956, 0.44405526, 0.26466702, 0.4461129,0.31844145, 0.38574855, 0.24185022, 0.47539435]


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

k_COMBINE_median_rank = [117,99,98,98,109,75,111,90,102,51,110,91,93,45,89,67,94,86,91,114,90,92,87,79,78,74,74,74,80,78,90,102]
s_COMBINE_median_rank = [35.0, 24.0, 48.0, 72.0, 33.0, 82.0, 71.0, 79.0, 31.0, 68.0, 70.0, 64.0, 27.0, 46.0, 74.0, 68.0, 38.0, 30.0, 59.0, 73.0, 38.0, 83.0, 70.0, 87.0, 32.0, 68.0, 70.0, 60.0, 30.0, 46.0, 75.0, 71.0]
p_COMBINE_median_rank = [112,63,122,110,110,58,115,108,109,57,106,105,112,72,112,97,77,69,75,84,78,75,83,83,76,75,85,87,80,80,84,83]
t_COMBINE_median_rank =[66.0, 67.0, 63.0, 64.0, 56.0, 41.0, 56.0, 57.0, 53.0, 36.0, 55.0, 54.0, 52.0, 33.0, 56.0, 54.0, 66.0, 59.0, 75.0, 89.0, 94.0, 80.0, 93.0, 97.0, 92.0, 81.0, 94.0, 92.0, 90.0, 80.0, 98.0, 102.0]
h_COMBINE_median_rank = [62.0, 103.0, 67.0, 63.0, 102.0, 53.0, 66.0, 98.0, 54.0, 60.0, 98.0, 56.0]
xt_COMBINE_median_rank = [47.0, 52.0, 54.0, 51.0, 47.0, 55.0, 54.0, 47.0, 48.0, 55.0, 60.0, 45.0, 42.0, 50.0, 54.0, 42.0]
cm_COMBINE_median_rank = [102.0, 105.0, 100.0, 99.0, 100.0, 104.0, 99.0, 97.0, 101.0, 103.5, 100.0, 98.0, 99.0, 105.0, 101.0, 102.0]
mc_COMBINE_median_rank = [48.0, 50.0, 52.0, 54.0, 47.0, 49.0, 48.0, 45.0, 50.0, 48.0, 54.0, 44.0, 38.0, 47.0, 48.0, 42.0]


COMBINE_median_rank = {}

COMBINE_median_rank['s'] = s_COMBINE_median_rank
COMBINE_median_rank['k'] = k_COMBINE_median_rank
COMBINE_median_rank['p'] = p_COMBINE_median_rank
COMBINE_median_rank['t'] = t_COMBINE_median_rank
COMBINE_median_rank['h'] = h_COMBINE_median_rank
COMBINE_median_rank['cm'] = cm_COMBINE_median_rank
COMBINE_median_rank['mc'] = mc_COMBINE_median_rank
COMBINE_median_rank['xt'] = xt_COMBINE_median_rank

k_COMBINE_median_proba = [0.39,0.42,0.43,0.42,0.4,0.6,0.4,0.44,0.41,0.79,0.4,0.44,0.29,0.85,0.44,0.52,0.36,0.36,0.45,0.34,0.40,0.31,0.41,0.47,0.47,0.50,0.5,0.5,0.46,0.38,0.37,0.17]
s_COMBINE_median_proba = [0.8340867, 1.1183634, 0.63963234, 0.5429022, 0.9156659, 0.4736061, 0.53365063, 0.49785916, 0.974964, 0.56085754, 0.55108014, 0.55968026, 1.11081101, 0.8231517, 0.51061213, 0.54890085, 0.804439, 0.8571037, 0.47703553, 0.5346647, 0.7860447, 0.47590486, 0.53243125, 0.4599289, 0.9605047, 0.56279995, 0.55249818, 0.5920737, 1.10010583, 0.8151276, 0.5154483, 0.5308167]
p_COMBINE_median_proba = [0.4,0.69,0.39,0.39,0.4,0.69,0.39,0.39,0.4,0.74,0.4,0.41,0.4,0.47,0.43,0.42,0.54,0.56,0.55,0.51,0.53,0.54,0.49,0.49,0.53,0.52,0.47,0.47,0.5,0.48,0.48,0.49]
t_COMBINE_median_proba = [0.5656828, 0.52959407, 0.5870697, 0.5679627, 0.6370576, 0.77518173, 0.60626706, 0.5856785, 0.6413826, 0.9007152, 0.64338623, 0.63190004, 0.67791357, 0.9935992, 0.6231174, 0.64545227, 0.56521106, 0.5812561, 0.5081865, 0.4313398, 0.34368709, 0.4297847, 0.3903978, 0.37568312, 0.3694328, 0.4311991, 0.3923239, 0.4004292, 0.41643158, 0.4066931, 0.2941942, 0.30279672]
h_COMBINE_median_proba = [0.4312863, 0.42057354, 0.5448893, 0.40214127, 0.4252485, 0.6699076, 0.36673294, 0.42175697, 0.661393, 0.3692991, 0.43237065, 0.61375718]
xt_COMBINE_median_proba = [0.63759703, 0.5977407, 0.59641385, 0.62374026, 0.64231576, 0.5838135, 0.58769872, 0.67730267, 0.61742757, 0.58565764, 0.54819565, 0.69354605, 0.70821126, 0.6400232, 0.583536, 0.72584325]

cm_COMBINE_median_proba = [0.4094048, 0.40615248, 0.41644475, 0.42815506, 0.4131622, 0.40594996, 0.4193576, 0.4313625, 0.4098963, 0.40927245, 0.41867783, 0.4261486, 0.41436898, 0.40508397, 0.41574463, 0.4202812]

mc_COMBINE_median_proba = [0.63508507, 0.6393516, 0.6178619, 0.5893003, 0.6543935, 0.63260067, 0.6569666, 0.69809835, 0.5959644, 0.6508962, 0.6090382, 0.695979, 0.7723243, 0.6767044, 0.6562135, 0.7310425]


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

s_MLP_median_rank = [57,39,78,73,49,82,78,88,47,71,82,71,50,65,81,80,65.0, 36.0, 52.0, 74.0, 49.0, 82.0, 79.0, 83.0, 50.0, 65.0, 76.0, 63.0, 48.0, 69.0, 82.5, 99.0]
k_MLP_median_rank = [104,104,98,95,116,87,118,97,110,62,118,94,77,63,111,85,68.0, 65.0, 83.0, 82.0, 85.0, 76.0, 75.0, 78.0, 75.0, 73.0, 100.0, 78.0, 82.0, 74.0, 88.0, 111.0]
p_MLP_median_rank = [119,70,122,119,114,66,120,114,115,63,113,111,115,65,104,80,88,68,63,76,121,112,124,100,96,76,112,105,121,98,116,86]
t_MLP_median_rank = [70,78,84,77,79,88,69,71,62,63,67,65,63,65,67,58,64.0, 71.0, 83.0, 103.0, 102.0, 112.0, 104.0, 110.0, 105.0, 109.5, 100.0, 111.5, 100.0, 103.0, 102.0, 114.0]
h_MLP_median_rank = [81,107,113,72,94,94,77,99,79,72,118,74,95,97,102,83]
xt_MLP_median_rank = [87,75,71,76,102,91,62,101,91,70,88,87,99,73,74,88]
cm_MLP_median_rank = [108,118,114,102,106,123,105,107,106,111,104,101,114,113,111,120]
mc_MLP_median_rank = [80,97,76,74,77,77,63,74,97,70,72,93,73,83,70,79]

MLP_median_rank = {}

MLP_median_rank['s'] = s_MLP_median_rank
MLP_median_rank['k'] = k_MLP_median_rank
MLP_median_rank['p'] = p_MLP_median_rank
MLP_median_rank['t'] = t_MLP_median_rank
MLP_median_rank['h'] = h_MLP_median_rank
MLP_median_rank['cm'] = cm_MLP_median_rank
MLP_median_rank['mc'] = mc_MLP_median_rank
MLP_median_rank['xt'] = xt_MLP_median_rank

s_MLP_median_proba = [0.61,0.79,0.37,0.58,0.70,0.50,0.52,0.46,0.72,0.58,0.53,0.57,0.71,0.77,0.56,0.53,0.53320643, 0.8452287, 0.68592196, 0.5680103, 0.6988786, 0.49760463, 0.51797, 0.4997376, 0.6983907, 0.57523176, 0.5542326, 0.6149157, 0.72050844, 0.73686037, 0.51554553, 0.4791451]
k_MLP_median_proba = [0.41,0.42,0.44,0.45,0.40,0.48,0.39,0.42,0.41,0.75,0.39,0.42,0.51,0.74,0.40,0.46,0.6117177, 0.61631007, 0.522272, 0.5368622, 0.52166344, 0.5470701, 0.5437947, 0.5304686, 0.5501571, 0.54854704, 0.4730489, 0.5392828, 0.5253682, 0.5122395, 0.44907173, 0.36036097]
p_MLP_median_proba = [0.39,0.68,0.38,0.38,0.40,0.70,0.40,0.40,0.40,0.71,0.39,0.40,0.40,0.67,0.41,0.48,0.53,0.59,0.56,0.55,0.52,0.33,0.17,0.47,0.51,0.57,0.40,0.44,0.24,0.47,0.36,0.51]
t_MLP_median_proba = [0.53,0.52,0.49,0.52,0.52,0.49,0.54,0.53,0.57,0.55,0.54,0.53,0.54,0.53,0.52,0.56,0.5903663, 0.5388082, 0.49992, 0.22984259, 0.12200762, 0.009, 0.21499847, 0.0306789, 0.065904885, 0.001, 0.29604826, 0.013696725, 0.1973622, 0.11121698, 0.27635708, 0.24985685]
h_MLP_median_proba = [0.53,0.43,0.31,0.57,0.45,0.50,0.53,0.44,0.59,0.54,0.41,0.62,0.45,0.45,0.43,0.53]
xt_MLP_median_proba = [0.50,0.52,0.55,0.51,0.44,0.48,0.61,0.44,0.47,0.55,0.40,0.51,0.43,0.53,0.55,0.50]
cm_MLP_median_proba = [0.42,0.4,0.4,0.43,0.36,0.42,0.42,0.42,0.41,0.42,0.43,0.41,0.4,0.41,0.4,0.41,0.4]
mc_MLP_median_proba = [0.52,0.45,0.52,0.51,0.54,0.52,0.61,0.53,0.45,0.55,0.54,0.47,0.55,0.50,0.56,0.52]

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
ax.scatter(CNN_rank, CNN_proba,c='blue', s=10,label = 'CNN')
points_cnn = np.array(list(zip(CNN_rank,CNN_proba)))
hull_cnn = ConvexHull(points_cnn)
for simplex in hull_cnn.simplices:
    ax.plot(points_cnn[simplex, 0], points_cnn[simplex, 1], 'b-')
ax.scatter(CNN_combine_rank, CNN_Combine_proba,c='green', s=10,label = 'CNN Combined')
ax.scatter([1],[1],c='black', s=100, label = 'Random guess')
points_cnn_combine = np.array(list(zip(CNN_combine_rank,CNN_Combine_proba)))
hull_cnn_combine = ConvexHull(points_cnn_combine)
for simplex in hull_cnn_combine.simplices:
    ax.plot(points_cnn_combine[simplex, 0], points_cnn_combine[simplex, 1], 'g-')
ax.scatter(MLP_rank, MLP_proba,c='red', s=10,label = 'MLP')

points_mlp = np.array(list(zip(MLP_rank,MLP_proba)))
hull_mlp = ConvexHull(points_mlp)
for simplex in hull_mlp.simplices:
    ax.plot(points_mlp[simplex, 0], points_mlp[simplex, 1], 'r-')
print('CNN consistency : ',hull_cnn.area)
print('Combined CNN consistency : ',hull_cnn_combine.area)
print('MLP consistency : ',hull_mlp.area)
ax.set_ylabel('Scores divided by random guess')
ax.set_xlabel('Ranks divided by random guess')
ax.legend()
ax.set_title('General consistency evaluation')
ax_sca = plt.gca()
ax_sca.set_xlim(0,1.1)
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