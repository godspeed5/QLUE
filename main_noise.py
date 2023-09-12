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
parser.add_argument('--output_dir', type=str, default='output/noise_sigma/', help='output directory')


args = parser.parse_args()

# Set directory
data_dir = args.dir
sortpar = args.sortpar
cq = args.cq
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)


noise_sizes = [0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750]
# dists=[100,200]
it = time.time()
iters = 30
sigmas = [1,5,10,100,1000]
h_scores = np.zeros((len(noise_sizes), iters, len(sigmas)))
# c_scores = np.zeros((len(noise_sizes), iters))
# v_scores = np.zeros((len(noise_sizes), iters))
for n_sigma, sigma in enumerate(sigmas):
    # print('Length is: ', len(glob.glob(data_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy')))
    if len(glob.glob(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy'))==0:
        for iter in range(iters):
            for noise_index in range(len(noise_sizes)):

                
                # wandb.run=None
                # run=wandb.init()
                cova = np.array([[sigma,0],[0,sigma]])

                # print(wandb.config)

                # dist  =  wandb.config.dist

                n_samples = [750]
                n_noise = noise_sizes[noise_index]

                means = [[0,0]]
                covs = [cova]
                prefactor = [1e5]*2
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

                df.to_csv(data_dir+ 'gen_data_'+str(noise_sizes[noise_index])+'_noise.csv')

                



                # Import the data for QLUE
                qlue_data = pd.read_csv(data_dir+ 'gen_data_'+str(noise_sizes[noise_index])+'_noise.csv')
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
                dataset_ld.to_csv('datasets/dataset_generated_ld_c_noise.csv')

                NH, dataset = calculateNearestHigher_classic_mod(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)

                # NH, dataset = calculateNearestHigher_classic_mod_hard(dataset_ld, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)
                # NH, dataset = calculateNearestHigher_classic_mod(dataset, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)    
                dataset.to_csv('datasets/dataset1_correct_generated_c_noise_circles.csv')
                dataset1 = findAndAssign_clusters_classic_fast(dataset, tileDict, delM)

                dataset1.to_csv('D1_correct_generated_c.csv')
                

                h_score,c_score,v_score = homogeneity_completeness_v_measure(dataset1['clusterId'].values, dataset1['ClusterNumbers'].values)
                print(h_score, c_score, v_score)
                h_scores[noise_index, iter,n_sigma] = h_score

                sns.scatterplot(data=dataset1, x="x", y="y", hue="ClusterNumbers", palette="deep").set_title('Computed clusters')
                plt.legend([],[], frameon=False)
                plt.ylim(-300,300)
                plt.text(0,275, 'score: '+ str(round(h_score,2)))
                plt.savefig(output_dir+'computed_clusters_'+str(noise_sizes[noise_index])+'_sigma_' + str(sigma)+'.png')
                plt.close()

                sns.scatterplot(data=dataset1, x="x", y="y", hue="clusterId", palette="deep").set_title('True clusters')
                plt.legend([],[], frameon=False)
                plt.savefig(output_dir+'true_clusters_'+str(noise_sizes[noise_index])+'_sigma_' + str(sigma)+'.png')
                plt.close()

        np.save(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy', h_scores)
    else:
        print('Loading from file')
        h_scores = np.load(output_dir+'h_scores_noise_sigma_' + str(sigma) + '.npy')          
    plt.plot([i/n_samples[0] for i in noise_sizes], np.mean(h_scores[:,:,n_sigma], axis=1), label='homogeneity')
    plt.fill_between(noise_sizes, np.mean(h_scores[:,:,n_sigma], axis=1)-np.std(h_scores[:,:,n_sigma], axis=1), np.mean(h_scores[:,:,n_sigma], axis=1)+np.std(h_scores[:,:,n_sigma], axis=1), alpha=0.2)
    # plt.savefig(output_dir+'homogeneity_noise_sigma_'+str(sigma)+'.png')
    # plt.close()
plt.xlabel('No. of noise samples/No. of cluster samples')
plt.ylabel('Homogeneity Score')
plt.legend(sigmas)
plt.savefig(output_dir+'homogeneity_noise_sigma.png')
plt.close()
            # v_scores[noise_index, iter] = v_score
    # print(h_scores)

# print(h_scores)
# print(c_scores)
# print(v_scores)

# np.save('h_scores_noise.npy', h_scores)
# np.save('c_scores_noise.npy', c_scores)
# np.save('v_scores_noise.npy', v_scores)

# h_scores = np.load('h_scores_noise.npy')
# c_scores = np.load('c_scores_noise.npy')
# v_scores = np.load('v_scores_noise.npy')
# print(h_scores.shape)
# print(c_scores.shape)
# print(v_scores.shape)
    

# plt.close()
# plt.plot(noise_sizes, np.mean(c_scores, axis=1), label='completeness')
# plt.fill_between(noise_sizes, np.mean(c_scores,axis=1)-np.std(c_scores, axis=1), np.mean(c_scores, axis=1)+np.std(c_scores, axis=1), alpha=0.2)
# plt.savefig(data_dir+'completeness_noise.png')
# plt.close()

# plt.plot(noise_sizes, v_scores, label='v_measure')
# plt.fill_between(noise_sizes, np.mean(v_scores, axis=1)-np.std(v_scores, axis=1), np.mean(v_scores, axis=1)+np.std(v_scores, axis=1), alpha=0.2)
# plt.savefig(data_dir+'v_measure_noise.png')
# plt.close()

# a = np.array([h_scores, c_scores, v_scores])
# np.save(data_dir+'all_scores_noise.npy', a)

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