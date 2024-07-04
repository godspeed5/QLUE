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
# wandb
import time
import glob
from energy_weighted_clustering_metrics import my_homogeneity_completeness_v_measure

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='datasets/')
parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
parser.add_argument('--cq', type=str, default='ch', help='classical or quantum or cheating')
parser.add_argument('--output_dir', type=str, default='outputs/overlap/plots3_nb/')


args = parser.parse_args()

# Set directory
data_dir = args.dir
sortpar = args.sortpar
cq = args.cq
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)
dists = [0,20,25,30,35,40,50,60,70,80,100,120,140]

iters = 30
h_scores = np.zeros((len(dists), iters))
c_scores = np.zeros((len(dists), iters))
v_scores = np.zeros((len(dists), iters))

it = time.time()
prefactors = [1,2,5,10]
ans = np.zeros((h_scores.shape[0], h_scores.shape[1], len(prefactors)))
for pfi, prefactor_factor in enumerate(prefactors):
    if len(glob.glob(output_dir+'h_score_'+str(prefactor_factor)+'.npy')) == 0:

        for iter in range(iters):

            for dist_index in range(len(dists)):

                ##### Create Gaussian clusters #####
                cova = np.array([[900,0],[0,900]])
                n_samples = [500, prefactor_factor*500]
                n_noise = 0

                means = [[-dists[dist_index],0],[dists[dist_index],0]]
                covs = [cova]*2
                prefactor = [500*2*np.pi*900, 500*2*np.pi*900]
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

                df.to_csv(data_dir+ 'gen_data_'+str(dists[dist_index])+'.csv')

                ######### Read data #########

                # Import the data for QLUE
                qlue_data = pd.read_csv(data_dir+ 'gen_data_'+str(dists[dist_index])+'.csv')
                # qlue_data = pd.read_csv(data_dir+ "dataset1.csv")

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

                # print('Density:', all(localDensities==trueDensity))
                dataset_ld.to_csv('datasets/dataset_generated_ld_overlap.csv')
                print('Local density calculated')

                NH, dataset = calculateNearestHigher_classic_mod(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)

                dataset1 = findAndAssign_clusters_classic_fast(dataset, tileDict, delM)

                h_score,c_score,v_score = my_homogeneity_completeness_v_measure(dataset1['clusterId'].values, dataset1['ClusterNumbers'].values, energy=dataset1['weight'].values)
                print(h_score, c_score, v_score)
                h_scores[dist_index, iter] = h_score
                c_scores[dist_index, iter] = c_score
                v_scores[dist_index, iter] = v_score

                sns.scatterplot(data=dataset1, x="x", y="y", hue="ClusterNumbers", edgecolor = "none", palette="deep").set_title('Computed clusters')
                plt.legend([],[], frameon=False)
                # plt.text(0,75,'score: ' + str(round(h_score, 2)))
                plt.ylim(-100,100)
                plt.xlim(-200,200)
                plt.text(-180,-80, '$\mathcal{F}_H= $'+ str(round(h_score,2)))
                plt.savefig(output_dir+'/computed_clusters_'+str(dists[dist_index])+'_'+str(prefactor_factor)+'.svg')
                plt.close()

                sns.scatterplot(data=dataset1, x="x", y="y", hue="clusterId", edgecolor = "none", palette="deep").set_title('True clusters')
                plt.legend([],[], frameon=False)
                plt.text(-180,-180, '$\mathcal{F}_H= $'+ str(round(h_score,2)))
                plt.savefig(output_dir + 'true_clusters_'+str(dists[dist_index])+'_'+str(prefactor_factor)+'.svg')
                plt.close()

        print(h_scores.shape)
        np.save(output_dir+'h_score_'+str(prefactor_factor)+'.npy', np.array(h_scores))
    else:
        print('Loading from file')
        h_scores = np.load(output_dir+'h_score_'+str(prefactor_factor)+'.npy')
    plt.plot([i/30 for i in dists[:10]], np.mean(h_scores[:10], axis=1), label='homogeneity_'+str((prefactor_factor)))
    plt.fill_between([i/30 for i in dists[:10]], np.mean(h_scores[:10], axis=1)-np.std(h_scores[:10], axis=1), np.mean(h_scores[:10], axis=1)+np.std(h_scores[:10], axis=1), alpha=0.2)
    # plt.savefig(output_dir+'homogeneity_varied_energy_'+str(pfi)+'.png')
    ans[:,:,pfi] = np.array(h_scores)
    
plt.xlabel('$N_{N}/N_{C}$')
plt.ylabel('$\mathcal{F}$')
# plt.xticks([0,0.5,1,1.5,2,2.5,3])
#plt.legend(['$\frac{N_1}{N_2}=$' + str(i) for i in prefactors])
plt.legend(['$N_1 / N_2=$' + str(i) for i in prefactors])
np.save(output_dir+'homogeneities_varied_energy.npy', ans)
# plt.text(100,0, h_score)
plt.savefig(output_dir+'homogeneity_varied_energy.svg')
plt.close()

ft = time.time()
print('Time taken:', ft-it)
