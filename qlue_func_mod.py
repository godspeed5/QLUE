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

def calculateLocalDensity_classic_mod(dataset, tileDict, dc, run_Grover = False):
    
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
        if run_Grover:
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
                # #print(fin_track)
            ##print(set(b)==set(i_fin_track), i)
        ##print(i)
        localDensities.append(temp_rho)
    dataset['rho'] = np.array(localDensities)
    return localDensities, dataset

def calculateNearestHigher_classic_mod(dataset, tileDict, dm_in, delC, rho_c, delM, run_grover=False):
    
    NHlist = list()
    tileIndices = tileDict.keys()
    Cnums = dict()
    Cnum = 0
    Onums = dict()
    deltas = np.zeros(len(dataset))
    # #print(dataset.head())

    # loop over all points
    for i in range(len(dataset)):
        temp_delta = math.inf
        NH_index = math.inf
        temp_rho = dataset['rho'][i] #initialize to the energy density of the point itself
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - dm_in, dataset.loc[i]['x'] + dm_in, dataset.loc[i]['y'] - dm_in, dataset.loc[i]['y'] + dm_in)
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
                    # #print(binData.head())

                    # iterate inside this bin
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if((dist <= dm_in) and (point['rho'] > temp_rho)):
                            if dist < temp_delta:
                                temp_delta = dist
                                NH_index = k #update nearest higher with current point
                                rho = point['rho']
                            elif dist == temp_delta and point['rho'] > rho: 
                                NH_index = k
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
                asasa=1
                #print('error')
                #print(fin_track)
                #print(a)
                #print('error')
        else:
            if len(fin_track) == 0:
                a = math.inf
            else:
                a = fin_track[0]
        if a in fin_track:
            NHlist.append(ai[a])
        else:
            NHlist.append(math.inf)
        # #print(a, fin_track)
        if NHlist[-1]!=math.inf:
            sdff = 234
            #print(temp_delta, distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]), temp_delta == distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]))
            asassa = 23
        
        if NH_index!=math.inf:
            #print(distance(dataset.loc[NH_index], dataset.loc[i]))
            delta = distance(dataset.loc[NH_index], dataset.loc[i])
            deltas[i] = delta
            #print(dataset['rho'][i], rho_c)
        else:
            delta = 999
            deltas[i] = 999
        if (delta > delC) and dataset['rho'][i] >= rho_c:
            Cnum +=1
            Cnums[i] = Cnum
        if delta > delM and dataset['rho'][i] < rho_c:
            Onums[i] = 1
        # else:

        ##print('Cnums: ', Cnums)
        ##print('Onums: ', Onums)
        # NHlist.append(NH_index)
        ##print(len(NHlist))
    dataset['NH'] = np.array(NHlist)
    Clist = np.zeros(len(dataset))
    Olist = np.zeros(len(dataset))
    isSeed = np.zeros(len(dataset))
    # deltas = np.zeros(len(dataset))
    for i in Cnums:
        Clist[i] = Cnums[i]
        isSeed[i] = 1
    for i in Onums:
        Olist[i] = Onums[i]

    ##print(Clist)
    dataset['ClusterNumbers'] = Clist
    if len(Onums) > 0:
        dataset['isOutlier'] = Olist
    dataset['isSeed'] = isSeed
    dataset['delta'] = deltas
    

    return NHlist, dataset

def calculateNearestHigher_classic_mod_hard(dataset, tileDict, dm_in, delC, rho_c, delM, run_grover=False):
    # delta_c = dc
    ##print('in nearest higher')
    NHlist = list()
    tileIndices = tileDict.keys()
    Cnums = dict()
    Cnum = 0
    Onums = dict()
    deltas = np.zeros(len(dataset))
    # #print(dataset.head())

    # loop over all points
    for i in range(len(dataset)):
        ##print('point number: ', i)
        temp_delta = dm_in
        d_low = 0
        NH_index = math.inf
        temp_rho = dataset['rho'][i] #initialize to the energy density of the point itself
        # get search box
        search_box = searchBox(dataset.loc[i]['x'] - temp_delta, dataset.loc[i]['x'] + temp_delta, dataset.loc[i]['y'] - temp_delta, dataset.loc[i]['y'] + temp_delta)
        # loop over bins in the search box
        indices =[]
        ai = []
        VI = []
        
        
        for xBin in range(search_box[0], search_box[1] + 1):
            for yBin in range(search_box[2], search_box[3] + 1):        
                # get the id of this bin
                binId = getGlobalBinByBin(xBin, yBin)
                # check if binId is in tileIndices
                if(binId in tileIndices):
                    # get points indices in dataset
                    dataIdx = tileDict[binId]
                    binData = dataset.loc[dataIdx]
                    # #print(binData.head())
                    
                    for k, point in binData.iterrows():
                        ai.append(k)
                        # query N_{dc}(i)
                        dist = distance(dataset.loc[i], point)
                        # sum weights within N_{dc}(i)
                        if (point['rho'] > temp_rho):
                            if dist < temp_delta and (dist>=d_low):
                                VI.append(k)
                    # fin_track = [9]
                    # fin_track_new = [3]
        #print('points satisfying bb before while: ', VI)
        c = 0 # X
        all_dists = [] # X
        justset = justhalf = -1
        while True: # X
            # #print('delta: ', temp_delta)
            # #print('d_low: ', d_low)
            # #print('VI: ', VI)
            # #print('justset: ', justset, 'justhalf: ', justhalf)
            # ##print('---------------------------------')
            c+=1 # X
            fin_track = [ai.index(i) for i in VI]
            apo = list(range(2*len(ai)+1))
            
            k = Grover(apo, fin_track, Printing=False)
            
            if k in fin_track:
                
                # for j in VI:
                #     ##print(distance(dataset.loc[j], dataset.loc[ai[k]]))
                #     ##print(dataset.loc[j])
                NH_index = k
                fin_track.remove(k)
                temp_delta = distance(dataset.loc[i], dataset.loc[ai[k]])
                all_dists.append(temp_delta) # X
                temp_delta = (temp_delta + d_low)/2 #X
                
                justset = 0 # X
                justhalf = 1 # X
                # ##print('delta in fin_track: ', temp_delta)
                # ##print('d_low in fin_track: ', d_low)
                
                # ##print('AI: ', ai)
                # ##print('VI: ', VI)
            else:
                
                # break

                if c==1 or justset == 1:
                    if NH_index != math.inf:
                        NH_index = ai[NH_index]
                    # ##print('dist: ', all_dists[-1], 'temp_delta: ', temp_delta)
                    # ##print('VI: ', VI)
                    ##print('NH: ', NH_index)
                    
                    break

                else:
                    # ##print('justhalf: ', justhalf)
                    justset = 1
                    justhalf = 0
                    d_low = temp_delta
                    # ##print('VI: ', VI)
                    # ##print('c: ', c)
                    temp_delta = all_dists[-1]

            
            VI = []
            for xBin in range(search_box[0], search_box[1] + 1):
                for yBin in range(search_box[2], search_box[3] + 1):        
                    # get the id of this bin
                    binId = getGlobalBinByBin(xBin, yBin)
                    # check if binId is in tileIndices
                    if(binId in tileIndices):
                        # get points indices in dataset
                        dataIdx = tileDict[binId]
                        binData = dataset.loc[dataIdx]
                        # ##print(binData.head())
                        
                        for k, point in binData.iterrows():
                            ai.append(k)
                            # query N_{dc}(i)
                            dist = distance(dataset.loc[i], point)
                            
                            # sum weights within N_{dc}(i)
                            if point['rho'] > temp_rho:
                                # ##print('dist: ', dist)
                                if dist < temp_delta and (dist>=d_low):
                                    VI.append(k)
        
        
        if NH_index != math.inf:
            for xBin in range(search_box[0], search_box[1] + 1):
                for yBin in range(search_box[2], search_box[3] + 1):        
                    # get the id of this bin
                    binId = getGlobalBinByBin(xBin, yBin)
                    # check if binId is in tileIndices
                    if(binId in tileIndices):
                        # get points indices in dataset
                        dataIdx = tileDict[binId]
                        binData = dataset.loc[dataIdx]
                        # ##print(binData.head())
                        
                        for k, point in binData.iterrows():
                            if distance(dataset.loc[i], dataset.loc[NH_index]) == distance(point, dataset.loc[i]):
                                if point['rho'] > dataset['rho'][NH_index]:
                                    NH_index = k
                                    ##print('in final loop: ', NH_index)

        
        
        if NH_index!=math.inf:
            ##print(distance(dataset.loc[NH_index], dataset.loc[i]))
            delta = distance(dataset.loc[NH_index], dataset.loc[i])
            deltas[i] = delta
            ##print(dataset['rho'][i], rho_c)
        else:
            delta = 999
            deltas[i] = 999
        if (delta > delC) and dataset['rho'][i] >= rho_c:
            Cnum +=1
            Cnums[i] = Cnum
        if delta > delM and dataset['rho'][i] < rho_c:
            Onums[i] = 1
        # else:

        ##print('Cnums: ', Cnums)
        ##print('Onums: ', Onums)
        NHlist.append(NH_index)
        # ##print(dms[-1])
    dataset['NH'] = NHlist
    Clist = np.zeros(len(dataset))
    Olist = np.zeros(len(dataset))
    isSeed = np.zeros(len(dataset))
    # deltas = np.zeros(len(dataset))
    for i in Cnums:
        Clist[i] = Cnums[i]
        isSeed[i] = 1
    for i in Onums:
        Olist[i] = Onums[i]

    ##print(Clist)
    dataset['ClusterNumbers'] = Clist
    if len(Onums) > 0:
        dataset['isOutlier'] = Olist
    dataset['isSeed'] = isSeed
    dataset['delta'] = deltas
    return NHlist, dataset

def findAndAssign_clusters_classic(dataset): #TODO: write faster function with nearest higher search box
    # tileIndices = tileDict.keys()
    window_size = max([distance(dataset.loc[dataset['NH'].loc[i]], dataset.loc[i]) for i in range(len(dataset)) if dataset['NH'].loc[i]!=math.inf])+2
        
    # iterate inside this bin
    seeds = dataset[dataset['ClusterNumbers']!=0]
    
    # ##print(seeds)
    for k_seed, point_seed in seeds.iterrows(): # loop over all seeds
        cluster = set([k_seed]) # list of points in cluster
        c = 1 # flag to check if cluster has changed
        a = 0
        indices =set()
        ai = set()
        while c: # while cluster has changed
            a+=1
            c=0 # reset flag
            old_cluster = len(cluster) # copy of cluster
            
            for k, point in dataset.iterrows(): # loop over all points in bin
                # ##print('before window check')
                window_check = any([distance(dataset.loc[i], point) <= window_size for i in cluster]) # boolean to check if any point in cluster is within the opened windows
                # ##print('after window check')
                if(k not in seeds and point['isOutlier']!=1) and window_check: # if point is not a seed and is not an outlier and is within the opened windows
                    ai.add(k) #append the index of the point to the list of points to search in
                    if point['NH'] in cluster: # point is a follower of any of the points in cluster
                        indices.add(k) # Append to the blackbox set for Grover
                        # if dataset['ClusterNumbers'].loc[k] != 0 and k not in seeds:
                        #     for i in cluster: #TODO: check merge logic
                        #         dataset['ClusterNumbers'].loc[i] = dataset['ClusterNumbers'].loc[k] #merge clusters
                        #         #print('merged')
                        dataset['ClusterNumbers'].loc[k] = point_seed['ClusterNumbers'] # assign cluster number to point
                        cluster.add(k) # add point to cluster
                        c=1 # set flag to 1
            #print(len(cluster))
            if len(cluster) == old_cluster:
                c=0
            #print('k_seed: ', k_seed, 'c: ', c, 'a: ', a)
            #print('ai: ', len(ai), 'indices: ', len(indices))
            #print(all([(i in ai) for i in indices]))
        
    return dataset

def findAndAssign_clusters_quantum(dataset):
    window_size = dataset['NH'].max()+2
    seeds = dataset[dataset['ClusterNumbers']!=0]
    for k_seed, point_seed in seeds.iterrows(): # loop over all seeds
        cluster = set([k_seed]) # list of points in cluster
        c = 1 # flag to check if cluster has changed
        a = 0
        indices =set()
        ai = set()
        while c: # while cluster has changed
            a+=1
            c=0 # reset flag
            old_cluster = len(cluster) # copy of cluster
            
            for k, point in dataset.iterrows(): # loop over all points in bin
                # #print('before window check')
                window_check = any([distance(dataset.loc[i], point) <= window_size for i in cluster]) # boolean to check if any point in cluster is within the opened wondows
                # #print('after window check')
                if(point['ClusterNumbers']==0 and point['isOutlier']!=1) and window_check: # if point is not a seed and is not an outlier and is within the opened windows
                    ai.add(k) #append the index of the point to the list of points to search in
                    if point['NH'] in cluster: # point is a follower of any of the points in cluster
                        indices.add(k) # Append to the blackbox set for Grover
                        dataset['ClusterNumbers'].loc[k] = point_seed['ClusterNumbers'] # assign cluster number to point
                        ai_list = list(ai)
                        fin_track = [ai_list.index(k)]
                        apo = list(range(2*len(ai)+1))
                        ans = Grover(apo, fin_track, Printing=False)
                        #print('k: ', k, 'ans: ', ans, 'ai[ans]: ', ai_list[ans])
                        cluster.add(ai_list[ans])
                        c=1 # set flag to 1
            #print(len(cluster))
            if len(cluster) == old_cluster:
                c=0
            
            #print('k_seed: ', k_seed, 'c: ', c, 'a: ', a)
            #print('ai: ', len(ai), 'indices: ', len(indices))
            #print(all([(i in ai_list) for i in indices]))
        
    return dataset

def findAndAssign_clusters_classic_fast(dataset, tileDict, dm_in): #TODO: write faster function with nearest higher search box
    # tileIndices = tileDict.keys()
    #print(dm_in)

    # window_size = max([distance(dataset.loc[dataset['NH'].loc[i]], dataset.loc[i]) for i in range(len(dataset)) if dataset['NH'].loc[i]!=math.inf])+2
        
    # iterate inside this bin
    seeds = dataset[dataset['isSeed']!=0] # list of seeds in dataset
    tileIndices = tileDict.keys() # list of all tiles
    #print(seeds)
    
    # #print(seeds)
    for k_seed, point_seed in seeds.iterrows(): # loop over all seeds
        cluster = set([k_seed]) # list of points in cluster
        c = 1 # flag to check if cluster has changed
        a = 0 # counter for iterations
        indices =set() # set of indices for blackbox
        ai = set() # set of indices to search in
        search_boxes = [] # list of search boxes
        while c: # while cluster has changed
            a+=1 # increment counter
            c=0 # reset flag
            old_cluster = len(cluster) # length of old cluster
            for i in cluster:
                search_boxes.append(searchBox(dataset.loc[i]['x'] - dm_in, dataset.loc[i]['x'] + dm_in, dataset.loc[i]['y'] - dm_in, dataset.loc[i]['y'] + dm_in)) # add search box to list
            curr_search_box = [min([i[0] for i in search_boxes]), max([i[1] for i in search_boxes]), min([i[2] for i in search_boxes]), max([i[3] for i in search_boxes])] # get current search box
            #print(curr_search_box)
            # #print('seed: ', dataset.loc[k_seed])
            # #print(search_boxes)
            # loop over bins in the search box
           
            # for i in range(len(search_boxes)): # loop over all search boxes            
            for xBin in range(curr_search_box[0], curr_search_box[1] + 1): 
                for yBin in range(curr_search_box[2], curr_search_box[3] + 1): # loop over all bins in search box
                    # get the id of this bin
                    binId = getGlobalBinByBin(xBin, yBin)
                    # #print(binId)
                    # check if binId is in tileIndices
                    if(binId in tileIndices):
                        # #print('binId: ', binId)
                        # get points indices in dataset
                        dataIdx = tileDict[binId]
                        # #print('dataIdx: ', dataIdx)
                        binData = dataset.loc[dataIdx]
                        # #print(binData.head())
                        # #print(binData.index)
                        for k, point in binData.iterrows(): # loop over all points in bin
                            # #print('NH: ', point['NH'] == k_seed)
                            if(k not in seeds and (not point['isOutlier'] if 'isOutlier' in point.keys() else True)): # if point is not a seed and is not an outlier
                                ai.add(k) #append the index of the point to the list of points to search in
                                if point['NH'] in cluster: # point is a follower of any of the points in cluster
                                    indices.add(k) # Append to the blackbox set for Grover
                                    dataset['ClusterNumbers'].loc[k] = point_seed['ClusterNumbers'] # assign cluster number to point
                                    cluster.add(k) # add point to cluster
                                    c=1 # set flag to 1  
                # exit()                  
            #print('cluster length: ', len(cluster))
            if len(cluster) == old_cluster:
                c=0
            #print('k_seed: ', k_seed, 'c: ', c, 'a: ', a)
            #print('ai: ', len(ai), 'indices: ', len(indices))
            #print(all([(i in ai) for i in indices]))
        
    return dataset