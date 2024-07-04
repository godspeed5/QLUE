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
import time
# import wandb
import glob
from energy_weighted_clustering_metrics import my_homogeneity_completeness_v_measure

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='datasets/')
parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
parser.add_argument('--cq', type=str, default='ch', help='classical or quantum or cheating')
parser.add_argument('--output_dir', type=str, default='outputs/noise_sigma/plots2_nb/', help='output directory')


args = parser.parse_args()

# Set directory
data_dir = args.dir
sortpar = args.sortpar
cq = args.cq
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)
n_samples = [750]
plt.rc('text', usetex=True)

noise_sizes = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750]
it = time.time()
iters = 30
sigmas = [3, 10, 32]

h_scores = np.zeros((len(noise_sizes), iters, len(sigmas)))
for n_sigma, sigma in enumerate(sigmas):
    if len(glob.glob(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy'))==0:
        for iter in range(iters):
            for noise_index in range(len(noise_sizes)):
                cova = np.array([[sigma**2,0],[0,sigma**2]])
                n_noise = noise_sizes[noise_index]
                means = [[0,0]]
                covs = [cova]
                prefactor = [2*np.pi*np.sqrt(np.linalg.det(covs))*500]*2
                n_clusters = len(means)
                df = pd.DataFrame(columns=['x','y','clusterId'])
                for i in range(n_clusters):
                    a = np.random.multivariate_normal(mean=means[i], cov=covs[i], size=n_samples[i])
                    df1 = pd.DataFrame(columns=['x','y','clusterId'], data=np.c_[a,(i+1)*np.ones((n_samples[i]))])
                    df1['weight'] = prefactor[i]*multivariate_normal.pdf(df1[['x', 'y']].values, mean=means[i], cov=covs[i])
                    df = pd.concat([df,df1])
                noise = np.random.uniform(-250,250, (n_noise,2))
                n = pd.DataFrame(columns=['x','y','clusterId', 'weight'], data=np.c_[noise,0*np.ones((n_noise)), np.random.random(n_noise)])
                df = pd.concat([df,n])
                df['layer']=np.zeros(len(df))

                df.to_csv(data_dir+ 'gen_data_'+str(noise_sizes[noise_index])+'_noise.csv')
                # Import the data for QLUE
                qlue_data = pd.read_csv(data_dir+ 'gen_data_'+str(noise_sizes[noise_index])+'_noise.csv')

                # Define variables and parameters needed by the algo
                outlierDeltaFactor = 2

                dc = 20
                rhoc = 25
                delM = outlierDeltaFactor*dc

                delC = dc
                phoC = rhoc

                # These variables can be modified and passed to functions in tiles.py
                #tilesMaxX = 250
                #tilesMinX = -250
                #tilesMaxY = 250
                #tilesMinY = -250
                #tileSize = 5
                #nColumns = math.ceil((tilesMaxX-tilesMinX)/(tileSize))
                #nRows =  math.ceil((tilesMaxY-tilesMinY)/(tileSize))

                # Take only data on selected layer
                chosen_layer = 0
                selected_data = qlue_data[qlue_data['layer']==chosen_layer].sort_values(sortpar, ascending=False, ignore_index=True)

                x = selected_data['x'].values
                y = selected_data['y'].values
                layer = selected_data['layer'].values
                weight = selected_data['weight'].values

                trueClusterId = selected_data['clusterId'].values

                # Create dataframe
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

                dataset_ld.to_csv('datasets/dataset_generated_ld_c_noise.csv')
                print('LDs computed')

                NH, dataset = calculateNearestHigher_classic_mod(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)

                dataset1 = findAndAssign_clusters_classic_fast(dataset, tileDict, delM)

                h_score,c_score,v_score = my_homogeneity_completeness_v_measure(dataset1['clusterId'].values, dataset1['ClusterNumbers'].values, energy=dataset1['weight'].values)
                print(h_score, c_score, v_score)
                h_scores[noise_index, iter,n_sigma] = h_score
                plt.rc('text', usetex=True)

                plt.tick_params(axis='y',which='both', left=True,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off)


                sns.scatterplot(data=dataset1, x="x", y="y", hue="ClusterNumbers", edgecolor = "none", palette="deep")
                plt.legend([],[], frameon=False)
                plt.ylim(-250,250)
                plt.xlim(-250,250)
                plt.text(-230,-230, '$\mathcal{F}_H= $'+ str(round(h_score,2)))
                plt.savefig(output_dir+'computed_clusters_'+str(noise_sizes[noise_index])+'_sigma_' + str(sigma)+'.svg')
                plt.close()

                sns.scatterplot(data=dataset1, x="x", y="y", hue="clusterId", edgecolor = "none", palette="deep").set_title('True clusters')
                plt.legend([],[], frameon=False)
                plt.savefig(output_dir+'true_clusters_'+str(noise_sizes[noise_index])+'_sigma_' + str(sigma)+'.svg')
                plt.close()

        np.save(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy', h_scores)
    else:
        print('Loading from file')
        h_scores = np.load(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy')
    print(h_scores.shape)
    plt.plot([i/n_samples[0] for i in noise_sizes], np.mean(h_scores[:,:,0], axis=1), label='homogeneity')
    plt.fill_between([i/n_samples[0] for i in noise_sizes], np.mean(h_scores[:,:,0], axis=1)-np.std(h_scores[:,:,0], axis=1), np.mean(h_scores[:,:,0], axis=1)+np.std(h_scores[:,:,0], axis=1), alpha=0.2)

plt.xlabel('$N_{N}/N_{C}$')
plt.ylabel('$\mathcal{F}$')
plt.legend(['$\sigma=$' + str(i) for i in sigmas])
plt.savefig(output_dir+'homogeneity_noise_sigma.svg')
plt.close()

ft = time.time()
print('Time taken:', ft-it)
