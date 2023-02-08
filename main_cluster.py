################################################################
################################################################
### ************MAIN FILE FOR QLUE Clustering*************** ###
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
import matplotlib.pyplot as plt
from copy import deepcopy
# from post_proc import *
from tiles import *
from qlue_func import *
from qlue_func_mod import *
from q_grover import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='datasets/')
    parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
    parser.add_argument('--cq', type=str, default='ch', help='classical or quantum or cheating')

    args = parser.parse_args()
    
    # Set directory
    data_dir = args.dir
    sortpar = args.sortpar
    cq = args.cq

    # Import the data for QLUE
    qlue_data = pd.read_csv(data_dir+ "dataset1.csv")
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
    selected_data = qlue_data[qlue_data['layer']==chosen_layer] #.sort_values(sortpar, ascending=False, ignore_index=True)

    x = selected_data['x'].values
    y = selected_data['y'].values
    layer = selected_data['layer'].values
    weight = selected_data['weight'].values

    trueDensity = selected_data['rho'].values
    trueNh = selected_data['NH'].values.astype(int)
    trueClusterNumber = selected_data['ClusterNumbers'].values
    trueiOutlier = selected_data['isOutlier'].values
    trueisSeed = selected_data['isSeed'].values
    trueDelta = selected_data['delta'].values
    # computedNH = selected_data['NH'].values.astype(int)
    # computedClusterNumbers = selected_data['ClusterNumbers'].values
    # computedisOutlier = selected_data['isOutlier'].values

    # Create dataframe
    # dataset = pd.DataFrame(np.array([x,y,layer,weight, trueNh, trueDensity, computedNH, computedClusterNumbers, computedisOutlier]).T, columns=['x','y','layer','weight', 'nh_old', 'rho', 'NH', 'ClusterNumbers', 'isOutlier'])
    dataset = pd.DataFrame(np.array([x,y,layer,weight, trueDensity, trueNh, trueClusterNumber, trueiOutlier, trueisSeed, trueDelta]).T, columns=['x','y','layer','weight','rho','NH','ClusterNumbers','isOutlier','isSeed','delta'])
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
        localDensities = calculateLocalDensity_classic_mod(dataset, tileDict, dc)
    elif cq == 'c':
        localDensities = calculateLocalDensity_classic(dataset, tileDict, dc)
    elif cq == 'ch':
        localDensities = trueDensity
    dataset['rho'] = localDensities
    # print(dataset.head())

    # print('Density:', all(localDensities==trueDensity))

    # NH, dataset = calculateNearestHigher_classic_mod_hard(dataset, tileDict, outlierDeltaFactor*dc, delC, phoC, delM)
    dataset1 = findAndAssign_clusters_classic_fast(dataset, tileDict, outlierDeltaFactor*dc)
    dataset1.to_csv('D1_fast_correct.csv')

    # print(dataset1[dataset1['isSeed']==1])
    # dataset1.to_csv('D1_toy.csv')
