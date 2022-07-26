##############################################
##############################################
### ******** FUNCTIONS FOR GROVER ******** ###
##############################################
##############################################


from cmath import log10
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import random
import math
from q_utilities import *
from scipy.special import gamma
from scipy.integrate import trapz
import sys
import argparse
import pandas as pd


################
### *************** QUBIT FORM OF A STATE *************** ###
################

### Determine the number of qubits required for a given dataset
def n_qubits(N):
    # Input: number of elements
    # Output: number of required qubits for Grover
    
    Dim = np.log2(N)

    if Dim == 0:
        Dim = 1

    Dim = np.ceil(Dim)
    
    return int(Dim)

### Quantum state for a single point (decimal to qubit)
def dec_to_qubit(index, data, form='qubit'):
    # Inputs 1: index of the point
    # Input 2: dataset fed into grover
    # Output: state vector (if form = "qubit") or dec array (if form = "dec")
    
    # Finding dimension:
    N = len(data)
    Dim = n_qubits(N)
    
    # Defining the state into its qubit form
    St = bin(int(index))[2:].zfill(Dim)[::-1]
    if (len(St) == 0):
        print("Error in defining the state!")
    if(form=='dec'):
        return St
    else:
        #tableau to be given into the state function
        tab = [np.mod(int(St[i])+1,2)*StD + np.mod(int(St[i]),2)*StU for i in range(len(St))]
        #state
        Stq = state(tab)
        
        return Stq

### Quantum state for a single point (qubit to decimal)
def qubit_to_dec(qstate, data, form = "dec"):
    # Inputs 1: state in the qubit form
    # Input 2: dataset fed into grover
    # Output: state corresponding in decimal form (if form = "dec") or cartesian form (if form = "cart")
    
    # Finding dimension:
    N = len(data)
    Dim = n_qubits(N)
    
    qubit_state = qstate
    
    # Find position of the biggest element in qstate
    index_max = int(qubit_state.argmax())
    # print(index_max)
    
    # Recover the dec form
    dec_form = bin(2**(Dim) - 1 - index_max)[2:].zfill(Dim)
    dec = int("0b"+dec_form[::-1],2)
      
    #return
    if form == "dec":
        return dec_form
    elif form == "cart":
        return data.loc[dec].to_numpy()



################
### TESTS ###
################
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='datasets/')

    args = parser.parse_args()

    myDir = args.dir

    dataset = pd.read_csv(myDir + "aniso_1000_20.00_25.00_2.00.csv")
    dataset = dataset[0:1000]
    print(dataset.head())
    print(len(dataset))

    index = random.randint(0,len(dataset))

    for index in range(len(dataset)):
        
        # print("RecHit info:\n")
        # print(dataset.loc[index].to_numpy())

        ######## Test qubit state 
        q_state = dec_to_qubit(index, dataset)
        # print("\nRecHit quantum state:")
        # print(q_state)

        ####### Test dec state
        dec_state = qubit_to_dec(q_state, dataset)
        # print("\nRecHit dec state:")
        # print(dec_state)

        cart_state = qubit_to_dec(q_state, dataset, form='cart')
        # print("\nRecHit cart state:")
        # print(cart_state)
        if((dataset.loc[index].to_numpy() != cart_state).all()):
            print("Failed\n")