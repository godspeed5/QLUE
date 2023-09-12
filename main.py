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
# wandb.init(project='qlue_overlap')

# Define sweep config
# sweep_configuration = {
#     'method': 'grid',
#     'name': 'sweep',
#     'parameters': 
#     {
#     'dist': {'values': [0,20,40,60,80,100]}
#     }
# }

# sweep_id = wandb.sweep(
#     sweep=sweep_configuration
#     )
# run = wandb.init()   





# if __name__ == "__main__":
    
### Create Gaussian clusters ###
# semiMajorAxis =50
# phi = 2*np.pi/3
# semiMinorAxis = 10
# varX1 = semiMajorAxis**2 * np.cos(phi)**2 + semiMinorAxis**2 * np.sin(phi)**2
# varX2 = semiMajorAxis** 2 * np.sin(phi)**2 + semiMinorAxis**2 * np.cos(phi)**2
# # cov12 = (semiMajorAxis**2 - semiMinorAxis**2) * np.sin(phi) * np.cos(phi) 
# cov12 = (2*np.random.random() -1)*np.sqrt(varX1*varX2)
# def main():

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='datasets/')
parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
parser.add_argument('--cq', type=str, default='ch', help='classical or quantum or cheating')

args = parser.parse_args()

# Set directory
data_dir = args.dir
sortpar = args.sortpar
cq = args.cq
output_dir = 'output_circles_num/'

os.makedirs(output_dir, exist_ok=True)
dists = [0,20,25,30,35,40,50,60,70,80,100,120,140]
iters = 30
h_scores = np.zeros((len(dists), iters))
c_scores = np.zeros((len(dists), iters))
v_scores = np.zeros((len(dists), iters))

# dists=[100,200]
it = time.time()
prefactors = [1,2,5,10,20]
ans = np.zeros((h_scores.shape[0], h_scores.shape[1], len(prefactors)))
for pfi, prefactor_factor in enumerate(prefactors):
    if len(glob.glob(output_dir+'h_score_'+str(pfi)+'.npy')) == 0:

        for iter in range(iters):

            for dist_index in range(len(dists)):

                ##### Create Gaussian clusters #####
                cova = np.array([[900,0],[0,900]])

                # print(wandb.config)

                # dist  =  wandb.config.dist



                n_samples = [500, prefactor_factor*500]
                n_noise = 0

                means = [[-dists[dist_index],0],[dists[dist_index],0]]
                covs = [cova]*2
                prefactor = [1e5, 1e5]
                n_clusters = len(means)
                df = pd.DataFrame(columns=['x','y','clusterId'])
                for i in range(n_clusters):
                    a = np.random.multivariate_normal(mean=means[i], cov=covs[i], size=n_samples[i])
                    df1 = pd.DataFrame(columns=['x','y','clusterId'], data=np.c_[a,(i+1)*np.ones((n_samples[i]))])
                    df1['weight'] = prefactor[i]*multivariate_normal.pdf(df1[['x', 'y']].values, mean=means[i], cov=covs[i])
                    df = pd.concat([df,df1])
                noise = np.random.uniform(-250,250, (n_noise,2))
                n = pd.DataFrame(columns=['x','y','clusterId', 'weight'], data=np.c_[noise,0*np.ones((n_noise)), 2*np.random.random(n_noise)])
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
                # delM = dc

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

                # trueDensity = selected_data['rho'].values
                # trueNh = selected_data['nh'].values.astype(int)
                # trueDelta = selected_data['delta'].values
                # trueisSeed = selected_data['isSeed'].values
                trueClusterId = selected_data['clusterId'].values


                # computedNH = selected_data['NH'].values.astype(int)
                # computedClusterNumbers = selected_data['ClusterNumbers'].values
                # computedisOutlier = selected_data['isOutlier'].values

                # Create dataframe
                # dataset = pd.DataFrame(np.array([x,y,layer,weight, trueNh, trueDensity, computedNH, computedClusterNumbers, computedisOutlier]).T, columns=['x','y','layer','weight', 'nh_old', 'rho', 'NH', 'ClusterNumbers', 'isOutlier'])
                dataset = pd.DataFrame(np.array([x,y,layer,weight, trueClusterId]).T, columns=['x','y','layer','weight', 'clusterId'])
                # Calculate tile indices and fill tiles as a dictionary
                tilesList = [getGlobalBin(x[k],y[k]) for k in range(len(x))]
                uniqueTileIdx, counts = np.unique(tilesList, return_counts=True)
                tileDict = {}

                for idx in uniqueTileIdx:
                    tileDict[idx] = np.where(tilesList==idx)[0]

                # dataset['tileIdx'] = tilesList
                # print(tileDict)
                # s = dec_to_qubit(3, dataset)
                # print(s)
                # print(qubit_to_dec(s, dataset))

                # TODO: Check if localdensities agree
                if cq == 'q':
                    ld, dataset_ld = calculateLocalDensity_classic_mod(dataset, tileDict, dc)
                elif cq == 'c':
                    ld, dataset_ld = calculateLocalDensity_classic(dataset, tileDict, dc)
                # elif cq == 'ch':
                #     localDensities = trueDensity
                # dataset['rho'] = localDensities
                # print(dataset.head())

                # print('Density:', all(localDensities==trueDensity))
                # dataset_ld.to_csv('datasets/dataset_generated_ld_c.csv')

                NH, dataset = calculateNearestHigher_classic_mod(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)

                # NH, dataset = calculateNearestHigher_classic_mod_hard(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)
                # NH, dataset = calculateNearestHigher_classic_mod(dataset, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)    
                # dataset.to_csv('datasets/dataset1_correct_generated_c.csv')
                dataset1 = findAndAssign_clusters_classic_fast(dataset, tileDict, delM)

                # dataset1.to_csv('D1_correct_generated_c.csv')
                

                h_score,c_score,v_score = homogeneity_completeness_v_measure(dataset1['clusterId'].values, dataset1['ClusterNumbers'].values)
                print(h_score, c_score, v_score)
                h_scores[dist_index, iter] = h_score
                c_scores[dist_index, iter] = c_score
                v_scores[dist_index, iter] = v_score

                sns.scatterplot(data=dataset1, x="x", y="y", hue="ClusterNumbers", palette="deep").set_title('Computed clusters')
                plt.legend([],[], frameon=False)
                plt.text(0,75,'score: ' + str(round(h_score, 2)))
                plt.ylim(-100,100)
                plt.xlim(-100,100)
                plt.savefig(output_dir+'/computed_clusters_'+str(dists[dist_index])+'_'+str(pfi)+'.png')
                plt.close()

                sns.scatterplot(data=dataset1, x="x", y="y", hue="clusterId", palette="deep").set_title('True clusters')
                plt.legend([],[], frameon=False)
                plt.savefig(output_dir + 'true_clusters_'+str(dists[dist_index])+'_'+str(pfi)+'.png')
                plt.close()

            # print(h_scores)

        # print(h_scores)
        # print(c_scores)
    #     # print(v_scores)
    # np.save('h_scores.npy', h_scores)
    # np.save('c_scores.npy', c_scores)
    # np.save('v_scores.npy', v_scores)

    # h_scores = np.load('h_scores.npy')
    # c_scores = np.load('c_scores.npy')
    # v_scores = np.load('v_scores.npy')
        print(h_scores.shape)
    # print(c_scores.shape)
    # print(v_scores.shape)
        # plt.plot(dists, np.mean(h_scores, axis=1), label='homogeneity_'+str((pfi)))
        # plt.fill_between(dists, np.mean(h_scores, axis=1)-np.std(h_scores, axis=1), np.mean(h_scores, axis=1)+np.std(h_scores, axis=1), alpha=0.2)
        # print(h_scores.shape, a.shape)
        np.save(output_dir+'h_score_'+str(pfi)+'.npy', np.array(h_scores))
    else:
        h_scores = np.load(output_dir+'h_score_'+str(pfi)+'.npy')
    plt.plot(dists, np.mean(h_scores, axis=1), label='homogeneity_'+str((pfi)))
    plt.fill_between(dists, np.mean(h_scores, axis=1)-np.std(h_scores, axis=1), np.mean(h_scores, axis=1)+np.std(h_scores, axis=1), alpha=0.2)
    # plt.savefig(output_dir+'homogeneity_varied_energy_'+str(pfi)+'.png')
    ans[:,:,pfi] = np.array(h_scores)
    
plt.xlabel('Distance between clusters')
plt.ylabel('Homogeneity')
plt.legend(prefactors)
np.save(output_dir+'homogeneities_varied_energy.npy', ans)
# plt.text(100,0, h_score)
plt.savefig(output_dir+'homogeneity_varied_energy.png')
plt.close()

# plt.close()
# plt.plot(dists, np.mean(c_scores, axis=1), label='completeness')
# plt.fill_between(dists, np.mean(c_scores,axis=1)-np.std(c_scores, axis=1), np.mean(c_scores, axis=1)+np.std(c_scores, axis=1), alpha=0.2)
# plt.savefig('completeness.png')
# plt.close()

# plt.plot(dists, v_scores, label='v_measure')
# plt.fill_between(dists, np.mean(v_scores, axis=1)-np.std(v_scores, axis=1), np.mean(v_scores, axis=1)+np.std(v_scores, axis=1), alpha=0.2)
# plt.savefig('v_measure.png')
# plt.close()



ft = time.time()
print('Time taken:', ft-it)

    # wandb.log({
    #     'homogeneity': h, 
    #     'completeness': c,
    #     'v_measure': v,
    #     })
# print(NH)
# print(dataset1.head())
# dataset1.to_csv('dataset1.csv')
# print(NH==trueNh)
# print(pauli_gen("I", 0, 2))


####### TODOLIST
#     # - calculate local density
    # write quantum version of the function using one of two methods (grover -> one-point state, grover -> all-points state)

    # - find nearest higher
    # use grover -> one-point state (good because nearest higher is unique)       

# wandb.agent(sweep_id, function=main)
# fig = plt.figure()
# plt.scatter(allX,allY)
# plt.savefig('scatter_plot.pdf')