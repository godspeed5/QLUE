################################################################
################################################################
### ***************** MAIN FILE FOR QLUE ******************* ###
################################################################
################################################################

import os
import sys
# from re import L
import numpy as np
import pandas as pd
import math
# from grover_op import *
import argparse
# from plot_utils import *

import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy
# from post_proc import *
from tiles import *
from qlue_func import *
from qlue_func_mod import *
from q_grover import *
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import *
from sklearn.metrics import *
matplotlib.use('Agg')
import wandb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='datasets/')
parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
parser.add_argument('--cq', type=str, default='ch', help='classical or quantum or cheating')
parser.add_argument('--dataset', type=str, default='moons', help='moons or circles or dataset_name')
parser.add_argument('--dc', type=int, default=20, help='dc')


args = parser.parse_args()

# Set directory
data_dir = args.dir
sortpar = args.sortpar
cq = args.cq
dataset_name = args.dataset
output_dir = 'output/'+dataset_name+'/'
dc = args.dc

it = time.time()


######### Read data #########

# Import the data for QLUE
qlue_data = pd.read_csv(data_dir+ dataset_name+'.csv')
# qlue_data = pd.read_csv(data_dir+ "dataset1.csv")

# Define variables and parameters needed by the algo
outlierDeltaFactor = 2

rhoc = 10
delM = outlierDeltaFactor*dc

delC = dc
phoC = rhoc

# Take only data on selected layer
chosen_layer = 0
selected_data = qlue_data[qlue_data['layer']==chosen_layer].sort_values(sortpar, ascending=False, ignore_index=True)

x = selected_data['x'].values
y = selected_data['y'].values
layer = selected_data['layer'].values
weight = selected_data['weight'].values

trueClusterId = selected_data['clusterId'].values


# Create dataframe
# dataset = pd.DataFrame(np.array([x,y,layer,weight, trueNh, trueDensity, computedNH, computedClusterNumbers, computedisOutlier]).T, columns=['x','y','layer','weight', 'nh_old', 'rho', 'NH', 'ClusterNumbers', 'isOutlier'])
dataset = pd.DataFrame(np.array([x,y,layer,weight, trueClusterId]).T, columns=['x','y','layer','weight', 'clusterId'])
# Calculate tile indices and fill tiles as a dictionary
tilesList = [getGlobalBin(x[k],y[k]) for k in range(len(x))]
uniqueTileIdx, counts = np.unique(tilesList, return_counts=True)
tileDict = {}

for idx in uniqueTileIdx:
    tileDict[idx] = np.where(tilesList==idx)[0]


# TODO: Check if localdensities agree
if cq == 'q':
    ld, dataset_ld = calculateLocalDensity_classic_mod(dataset, tileDict, dc)
elif cq == 'c':
    ld, dataset_ld = calculateLocalDensity_classic(dataset, tileDict, dc)

dataset1 = pd.read_csv(output_dir+'D1_correct_generated_c.csv')

h_score,c_score,v_score = homogeneity_completeness_v_measure(dataset1['clusterId'].values, dataset1['ClusterNumbers'].values)
print(h_score, c_score, v_score)
np.array([h_score, c_score, v_score]).tofile(output_dir+'scores.csv', sep=',')

plt.rc('text', usetex=True)

sns.scatterplot(data=dataset1, x="x", y = 'y', hue="ClusterNumbers", palette="deep", linewidth=0).set(xlabel='x',ylabel=None)
plt.tick_params(axis='x',which='both', bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off)
plt.legend([],[], frameon=False)
plt.text(-100,-50,'$\mathcal{F}_H=$' + str(round(h_score, 2)))
plt.text(-100,-30,'$\mathcal{F}_C=$' + str(round(c_score, 2)))
plt.savefig(output_dir+'computed_clusters.svg')

plt.close()

sns.scatterplot(data=dataset1, x="x", y='y', hue="clusterId", palette="deep")
plt.legend([],[], frameon=False)
plt.savefig(output_dir+'true_clusters.svg')
plt.close()

ft = time.time()
print('Time taken:', ft-it)