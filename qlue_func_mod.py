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
    for i in range(len(dataset[0:20])):
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
        if NHlist[-1]!=math.inf:
            print(temp_delta, distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]), temp_delta == distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]))
            # print(rho, dataset['rho'].iloc[NHlist[-1]], rho == dataset['rho'].iloc[NHlist[-1]])
            # print('\n')
            # print(dataset.iloc[i].to_numpy())
            # print(dataset.iloc[NHlist[-1]].to_numpy())
            # print(dataset.iloc[int(dataset['nh'].iloc[i])].to_numpy())
            # print('\n')
    return NHlist

def calculateNearestHigher_classic_mod_hard(dataset, tileDict, dm_in, dc, rho_c, delM, run_grover=False):
    # delta_c = dc

    NHlist = list()
    tileIndices = tileDict.keys()
    Cnums = dict()
    Cnum = 0
    Onums = dict()
    # print(dataset.head())

    # loop over all points
    for i in range(len(dataset)):
        print('point number: ', i)
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
                    # print(binData.head())
                    
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
        print('points satisfying bb before while: ', VI)
        c = 0 # X
        all_dists = [] # X
        justset = justhalf = -1
        while True: # X
            # print('delta: ', temp_delta)
            # print('d_low: ', d_low)
            # print('VI: ', VI)
            # print('justset: ', justset, 'justhalf: ', justhalf)
            # print('---------------------------------')
            c+=1 # X
            fin_track = [ai.index(i) for i in VI]
            apo = list(range(2*len(ai)+1))
            
            k = Grover(apo, fin_track, Printing=False)
            
            if k in fin_track:
                
                # for j in VI:
                #     print(distance(dataset.loc[j], dataset.loc[ai[k]]))
                #     print(dataset.loc[j])
                NH_index = k
                fin_track.remove(k)
                temp_delta = distance(dataset.loc[i], dataset.loc[ai[k]])
                all_dists.append(temp_delta) # X
                temp_delta = (temp_delta + d_low)/2 #X
                
                justset = 0 # X
                justhalf = 1 # X
                # print('delta in fin_track: ', temp_delta)
                # print('d_low in fin_track: ', d_low)
                
                # print('AI: ', ai)
                # print('VI: ', VI)
            else:
                
                # break

                if c==1 or justset == 1:
                    if NH_index != math.inf:
                        NH_index = ai[NH_index]
                    # print('dist: ', all_dists[-1], 'temp_delta: ', temp_delta)
                    # print('VI: ', VI)
                    print('NH: ', NH_index)
                    
                    break

                else:
                    # print('justhalf: ', justhalf)
                    justset = 1
                    justhalf = 0
                    d_low = temp_delta
                    # print('VI: ', VI)
                    # print('c: ', c)
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
                        # print(binData.head())
                        
                        for k, point in binData.iterrows():
                            ai.append(k)
                            # query N_{dc}(i)
                            dist = distance(dataset.loc[i], point)
                            
                            # sum weights within N_{dc}(i)
                            if point['rho'] > temp_rho:
                                # print('dist: ', dist)
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
                        # print(binData.head())
                        
                        for k, point in binData.iterrows():
                            if distance(dataset.loc[i], dataset.loc[NH_index]) == distance(point, dataset.loc[i]):
                                if point['rho'] > dataset['rho'][NH_index]:
                                    NH_index = k
                                    print('in final loop: ', NH_index)

        
        
        if NH_index!=math.inf:
            print(distance(dataset.loc[NH_index], dataset.loc[i]))
            delta = distance(dataset.loc[NH_index], dataset.loc[i])
            print(dataset['rho'][i], rho_c)
            if delta > dc and dataset['rho'][i] > rho_c:
                Cnum +=1
                Cnums[i] = Cnum
            if delta > delM and dataset['rho'][i] < rho_c:
                Onums[i] = 1
        print('Cnums: ', Cnums)
        print('Onums: ', Onums)
        NHlist.append(NH_index)
        # print(dms[-1])
    dataset['NH'] = NHlist
    Clist = np.zeros(len(dataset))
    Olist = np.zeros(len(dataset))
    
    for i in Cnums:
        Clist[i] = Cnums[i]
    for i in Onums:
        Olist[i] = Onums[i]

    print(Clist)
    dataset['ClusterNumbers'] = Clist
    dataset['isOutlier'] = Olist
    return NHlist, dataset
    
                    





                    # print('in if2')
                    
                    
                    

                    


                    # while len(apo)

                    # iterate inside this bin
                    # for k, point in binData.iterrows():
                    #     # query N_{dc}(i)
                    #     dist = distance(dataset.loc[i], point)
                    #     # sum weights within N_{dc}(i)
                    #     if((dist <= dm) and (point['rho'] > temp_rho)):
                    #         if dist <= temp_delta:
                    #             temp_delta = dist
                    #             NH_index = k #update nearest higher with current point
                    #             rho = point['rho']
                    #             # bestpoint = point

                    #          #append the index of the point
                    # if NH_index != math.inf:
                    #     indices = [NH_index]

        # fin_track = [ai.index(i) for i in indices]          # indices of the points that satisfy the condition

        # list of all possible tracksters (increase length if more than half the points satisfy the condition)
        
        # if run_grover:            
        #     if len(fin_track)>=len(ai)//2:
                # apo = list(range(2*len(ai)+1))              
        #     else:
        #         apo = list(range(len(ai)))
        #     a = Grover(apo, fin_track, Printing=False)
        #     if len(fin_track)!=0 and [a]!=fin_track:
        #         print('error')
        #         print(fin_track)
        #         print(a)
        #         print('error')
        # else:
        #     if len(fin_track) == 0:
        #         a = math.inf
        #     else:
        #         a = fin_track[0]
        # if a in fin_track:
        #     NHlist.append(ai[a])
        # else:
        #     NHlist.append(math.inf)
        # print(a, fin_track)
        # if NHlist[-1]!=math.inf:
            # print(temp_delta, distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]), temp_delta == distance(dataset.iloc[NHlist[-1]], dataset.iloc[i]))
            # print(rho, dataset['rho'].iloc[NHlist[-1]], rho == dataset['rho'].iloc[NHlist[-1]])
            # print('\n')
            # print(dataset.iloc[i].to_numpy())
            # print(dataset.iloc[NHlist[-1]].to_numpy())
            # print(dataset.iloc[int(dataset['nh'].iloc[i])].to_numpy())
            # print('\n')
    

def findAndAssign_clusters_classic(dataset, tileDict, dm):
    
    localDensities = list()
    tileIndices = tileDict.keys()
    window_size = dataset.NH.max()
    dataset['clusterNumber'] = np.zeros(len(dataset))

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
                    seeds = binData[binData['isSeed']==1]
                    for k_seed, point_seed in seeds.iterrows(): # loop over all seeds
                        point_seed['clusterNumber'] = k_seed # assign cluster number to seed
                        cluster = [k_seed] # list of points in cluster
                        c = 1 # flag to check if cluster has changed
                        while c: # while cluster has changed
                            c=0 # reset flag
                            old_cluster = cluster.copy() # copy of cluster
                            for k, point in binData.iterrows(): # loop over all points in bin
                                window_check = any([distance(dataset.loc[i], point) <= window_size for i in cluster]) # boolean to check if any point in cluster is within the opened wondows
                                if(point['isSeed']!=1 and point['isOutlier']!=1) and window_check: # if point is not a seed and is not an outlier and is within the opened windows
                                    ai.append(k) #append the index of the point to the list of points to search in
                                if point['NH'] in cluster: # point is a follower of any of the points in cluster
                                    indices.append(k) # Append to the blackbox set for Grover
                                    point['clusterNumber'] = k_seed # assign cluster number to point
                                    cluster.append(k) # add point to cluster
                                    c=1 # set flag to 1

                                    
                        
        
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