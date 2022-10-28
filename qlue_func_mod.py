import numpy as np
from tiles import *
from q_grover import Grover

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
                            indices.append(k)
                        elif(dist < 1e-8):
                            temp_rho += point['weight']
        
        fin_track = [ai.index(i) for i in indices]
        i_fin_track = [ai.index(i) for i in indices]
        # print(i_fin_track)
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