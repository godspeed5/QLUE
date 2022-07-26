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
# import argparse
# from plot_utils import *
import matplotlib.pyplot as plt
from copy import deepcopy
# from post_proc import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='datasets/')
    parser.add_argument('--sortpar', type=str, default='weight', help='weight or rho')
    # parser.add_argument('--en', type=float, default='0.6')
    # parser.add_argument('--encum', type=float, default='0.5')
    # parser.add_argument('--pval', type=float, default='0.99')

    # args = parser.parse_args()
    
    # Set directory
    data_dir = args.dir
    sortpar = args.sortpar

    # Import the data for QLUE
    qlue_data = pd.read_csv(data_dir+ "aniso_1000_20.00_25.00_2.00.csv")

    chosen_layer = 10

    # Take only data on selected layer
    selected_data = qlue_data[qlue_data['layer']==chosen_layer].sort_values(sortpar, ascending=False, ignore_index=True)

    allX = selected_data['x'].values
    allY = selected_data['y'].values
    allLayer = selected_data['layer'].values
    allWeight = selected_data['weight'].values
    allDensity = selected_data['rho'].values

    # print(selected_data)

    
    # fig = plt.figure()
    # plt.scatter(allX,allY)
    # plt.savefig('scatter_plot.pdf')

