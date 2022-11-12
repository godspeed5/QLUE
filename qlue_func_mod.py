from ctypes.wintypes import FLOAT
import numpy as np
from tiles import *
from q_grover import Grover
import sys
import math

def distance(data_i, data_j):
    dx = data_i['x'] - data_j['x']
    dy = data_i['y'] - data_j['y']
    return np.sqrt(dx**2 + dy**2)

def calculateLocalDensity_classic_mod(dataset, tileDict, dc):
    
    localDensities = list()
    tileIndices = tileDict.keys()
    # loop over all points
    for i in range(len(dataset)):
        temp_rho = 0
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - dc, dataset.loc[i]['x'] + dc, dataset.loc[i]['y'] - dc, dataset.loc[i]['y'] + dc)
        # loop over bins in the search box
        indices =[]
        ai = []
        for xBin in range(search_box[0], search_box[1] + 1):
            for yBin in range(search_box[2], search_box[3] + 1):        
                # get the id of this bin
                binId = getGlobalBinByBin(xBin, yBin)
                # check if binId is in tileIndices
                if(binId in tileIndices):
                    # get points indices in dataset
                    dataIdx = tileDict[binId]
                    binData = dataset.loc[dataIdx]
                    

                    # iterate inside this bin
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if((dist <= dc) and (dist > 1e-8)):
                            temp_rho += 0.5 * point['weight']
                            indices.append(k) #append the index of the point
                        elif(dist < 1e-8):
                            temp_rho += point['weight']
        
        fin_track = [ai.index(i) for i in indices]          # indices of the points that satisfy the condition
        i_fin_track = [ai.index(i) for i in indices]        #copy of fin_tracK

        # list of all possible tracksters (increase length if more than half the points satisfy the condition)
        if len(fin_track)>=len(ai)//2:
            apo = list(range(2*len(ai)+1))              
        else:
            apo = list(range(len(ai)))                  
        b = []
        a = Grover(apo, fin_track, Printing=False)
        while a in fin_track:
            fin_track.remove(a)
            b.append(a)
            a = Grover(apo, fin_track, Printing=False)
            print(fin_track)            
        print(set(b)==set(i_fin_track))

        localDensities.append(temp_rho)
    return localDensities

def calculateNearestHigher_classic_mod(dataset, tileDict, dm, run_grover=False):
    
    NHlist = list()
    tileIndices = tileDict.keys()
    # print(dataset.head())

    # loop over all points
    for i in range(len(dataset)):
        temp_delta = math.inf
        NH_index = math.inf
        temp_rho = dataset['rho'][i] #initialize to the energy density of the point itself
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - dm, dataset.loc[i]['x'] + dm, dataset.loc[i]['y'] - dm, dataset.loc[i]['y'] + dm)
        # loop over bins in the search box
        indices =[]
        ai = []
        for xBin in range(search_box[0], search_box[1] + 1):
            for yBin in range(search_box[2], search_box[3] + 1):        
                # get the id of this bin
                binId = getGlobalBinByBin(xBin, yBin)
                # check if binId is in tileIndices
                if(binId in tileIndices):
                    # get points indices in dataset
                    dataIdx = tileDict[binId]
                    binData = dataset.loc[dataIdx]
                    # print(binData.head())

                    # iterate inside this bin
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if((dist <= dm) and (point['rho'] > temp_rho)):
                            if dist <= temp_delta:
                                temp_delta = dist
                                NH_index = k #update nearest higher with current point
                                rho = point['rho']
                                # bestpoint = point

                             #append the index of the point
                    if NH_index != math.inf:
                        indices = [NH_index]

        fin_track = [ai.index(i) for i in indices]          # indices of the points that satisfy the condition

        # list of all possible tracksters (increase length if more than half the points satisfy the condition)
        
        if run_grover:            
            if len(fin_track)>=len(ai)//2:
                apo = list(range(2*len(ai)+1))              
            else:
                apo = list(range(len(ai)))
            a = Grover(apo, fin_track, Printing=False)
            if len(fin_track)!=0 and [a]!=fin_track:
                print('error')
                print(fin_track)
                print(a)
                print('error')
        else:
            if len(fin_track) == 0:
                a = math.inf
            else:
                a = fin_track[0]
        if a in fin_track:
            NHlist.append(ai[a])
        else:
            NHlist.append(math.inf)
        # print(a, fin_track)
        # if NHlist[-1]!=math.inf:
            # print(temp_delta, distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]), temp_delta == distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]))
            # print(rho, dataset['rho'].iloc[NHlist[-1]], rho == dataset['rho'].iloc[NHlist[-1]])
            # print('\n')
            # print(dataset.iloc[i].to_numpy())
            # print(dataset.iloc[NHlist[-1]].to_numpy())
            # print(dataset.iloc[int(dataset['nh'].iloc[i])].to_numpy())
            # print('\n')
    return NHlist

def calculateNearestHigher_classic_mod_hard(dataset, tileDict, dm, run_grover=False):
    
    NHlist = list()
    tileIndices = tileDict.keys()
    # print(dataset.head())

    # loop over all points
    for i in range(len(dataset)):
        temp_delta = math.inf
        NH_index = math.inf
        temp_rho = dataset['rho'][i] #initialize to the energy density of the point itself
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - dm, dataset.loc[i]['x'] + dm, dataset.loc[i]['y'] - dm, dataset.loc[i]['y'] + dm)
        # loop over bins in the search box
        indices =[]
        ai = []
        for xBin in range(search_box[0], search_box[1] + 1):
            for yBin in range(search_box[2], search_box[3] + 1):        
                # get the id of this bin
                binId = getGlobalBinByBin(xBin, yBin)
                # check if binId is in tileIndices
                if(binId in tileIndices):
                    # get points indices in dataset
                    dataIdx = tileDict[binId]
                    binData = dataset.loc[dataIdx]
                    # print(binData.head())

                    # iterate inside this bin
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if((dist <= dm) and (point['rho'] > temp_rho)):
                            if dist <= temp_delta:
                                temp_delta = dist
                                NH_index = k #update nearest higher with current point
                                rho = point['rho']
                                # bestpoint = point

                             #append the index of the point
                    if NH_index != math.inf:
                        indices = [NH_index]

        fin_track = [ai.index(i) for i in indices]          # indices of the points that satisfy the condition

        # list of all possible tracksters (increase length if more than half the points satisfy the condition)
        
        if run_grover:            
            if len(fin_track)>=len(ai)//2:
                apo = list(range(2*len(ai)+1))              
            else:
                apo = list(range(len(ai)))
            a = Grover(apo, fin_track, Printing=False)
            if len(fin_track)!=0 and [a]!=fin_track:
                print('error')
                print(fin_track)
                print(a)
                print('error')
        else:
            if len(fin_track) == 0:
                a = math.inf
            else:
                a = fin_track[0]
        if a in fin_track:
            NHlist.append(ai[a])
        else:
            NHlist.append(math.inf)
        # print(a, fin_track)
        # if NHlist[-1]!=math.inf:
            # print(temp_delta, distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]), temp_delta == distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]))
            # print(rho, dataset['rho'].iloc[NHlist[-1]], rho == dataset['rho'].iloc[NHlist[-1]])
            # print('\n')
            # print(dataset.iloc[i].to_numpy())
            # print(dataset.iloc[NHlist[-1]].to_numpy())
            # print(dataset.iloc[int(dataset['nh'].iloc[i])].to_numpy())
            # print('\n')
    return NHlist

def findAndAssign_clusters_classic_mod_hard(dataset, tileDict, dm):
    
    localDensities = list()
    tileIndices = tileDict.keys()

    # loop over all points
    for i in range(len(dataset)):
        temp_delta = math.inf
        temp_rho = 0
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - dm, dataset.loc[i]['x'] + dm, dataset.loc[i]['y'] - dm, dataset.loc[i]['y'] + dc)
        # loop over bins in the search box
        indices =[]
        ai = []
        for xBin in range(search_box[0], search_box[1] + 1):
            for yBin in range(search_box[2], search_box[3] + 1):        
                # get the id of this bin
                binId = getGlobalBinByBin(xBin, yBin)
                # check if binId is in tileIndices
                if(binId in tileIndices):
                    # get points indices in dataset
                    dataIdx = tileDict[binId]
                    binData = dataset.loc[dataIdx]
                    

                    # iterate inside this bin
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if((dist <= dm) and (dist > 1e-8)):
                            if dist <= temp_delta:
                                temp_delta = dist
                                # wut
                            indices.append(k) #append the index of the point
                        elif(dist < 1e-8):
                            temp_rho += point['weight']
        
        fin_track = [ai.index(i) for i in indices]          # indices of the points that satisfy the condition
        i_fin_track = [ai.index(i) for i in indices]        #copy of fin_tracK

        # list of all possible tracksters (increase length if more than half the points satisfy the condition)
        if len(fin_track)>=len(ai)//2:
            apo = list(range(2*len(ai)+1))              
        else:
            apo = list(range(len(ai)))                  
        b = []
        a = Grover(apo, fin_track, Printing=False)
        while a in fin_track:
            fin_track.remove(a)
            b.append(a)
            a = Grover(apo, fin_track, Printing=False)
            print(fin_track)            
        print(set(b)==set(i_fin_track))

        localDensities.append(temp_rho)
    return localDensities