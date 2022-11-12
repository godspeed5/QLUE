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
import itertools


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
    # print(index)
    
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


def l_sup(all_points_ordered, input_form = "dec", output_form = "qubit"):
    # Input 1: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 2: dataset fed into grover
    # Input 3: input form - currently is ONLY "dec"
    # Input 4: output form - choose whether outputting the state in "dec" or "qubit" form
    # Output: superposition state in qubit form
    
    #Find and return all possible tracksters 
    all_tracksters = all_points_ordered
    
    if output_form == "dec":
        return all_tracksters
    elif output_form == "qubit":
        # print([dec_to_qubit(trackster, all_points_ordered).shape for trackster in all_tracksters])
        return np.sum([dec_to_qubit(trackster, all_points_ordered) for trackster in all_tracksters])/np.sqrt(len(all_tracksters))

def remove_array_from_list(full_list, list_to_remove):
    for index in range(len(full_list)):
        if np.array_equal(full_list[index], list_to_remove):
            full_list.pop(index)
            break

def black_box(full_state, fin_track, input_form = "dec", output_form = "Operator", Printing = False):
    # Input 1: thresholds to be used. First one is when there are no missing layers in between, second one when there are. 
    # Input 2: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 3: dataset fed into grover
    # Input 4: input form - currently is ONLY "dec"
    # Input 5: output form - Decides whether spit out the operator ()"Operator") or the list of the points that have been found ("List")
    # Output: operator that gives a phase every time the thresholds are satisfied
    #Find all possible tracksters in dec form:
    all_tracksters_dec = l_sup(full_state, input_form = input_form, output_form = "dec")
    dimQ = n_qubits(len(full_state))
    
    #If there are no points left, return the identity
    if len(all_tracksters_dec) == 0:
        if output_form == "Operator":
            return pauli_gen("I" , 0 , dimQ)
        elif output_form == "List":
            return []

    # Build the desired discerning operator:
    if len(fin_track) == 0:
        if output_form == "Operator":
            return pauli_gen("I" , 0 , dimQ)
        elif output_form == "List":
            return fin_track
    
    else: 
        #States to be marked:
        st_to_mark = np.sum([dec_to_qubit(track, full_state) for track in fin_track])/np.sqrt(len(fin_track))
        #st_to_mark = [full_dec_to_qubit(track, dataset) for track in fin_track]
        #op_proj = np.sum([st_proj(st,st) for st in st_to_mark])
        if output_form == "Operator":
            bb_op = pauli_gen("I" , 0 , dimQ) - 2 * st_proj(st_to_mark,st_to_mark)
            
            #bb_op = pauli_gen("I" , 0 , len(dataset)) - 2 * op_proj
            if Printing:
                print("Black box operator:")
                print("Unitarity of the black box operator (0 is good):", abs(bb_op.conjugate().transpose().dot(bb_op) - pauli_gen("I" , 0 , dimQ)).max())
            return bb_op
        elif output_form == "List":
            return fin_track

### Grover routine
def Grover(all_points_ordered, fin_track, input_form = "dec", output_form = "dec", Printing = False):
    # Input 2: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 3: dataset fed into grover
    # Input 4: input form - currently is ONLY "dec"
    # Input 5: output form - Decides whether spit out the qubit state ("qubit") or the dec state ("dec")
    # Output: state found by grover

    dimQ = n_qubits(len(all_points_ordered))
    
    #Do the Grover search until we find all points:
    #Find all possible tracksters:
    all_tracksters_dec = l_sup(all_points_ordered, input_form = input_form, output_form = "dec")
    all_tracksters_qubit = l_sup(all_points_ordered, input_form = input_form, output_form = "qubit")
    
    #Define the relevant operators:
    #black box
    bb = black_box(all_points_ordered, fin_track, input_form = input_form, output_form = "Operator", Printing = Printing)
    # print("fin_track", fin_track)

    #state projector
    st_proj_op = 2*st_proj(all_tracksters_qubit,all_tracksters_qubit) - pauli_gen("I" , 0 , dimQ)
    if Printing:
        print("Grover box: number of points still to be found:", len(fin_track))
        print("Unitarity of the state projector (0 is good): ", abs(st_proj_op.conjugate().transpose().dot(st_proj_op) - pauli_gen("I" , 0 , dimQ)).max())
    
    #Number of iterations:
    if len(fin_track) != 0:
        Niter = np.round(np.pi / (4 * np.arcsin(np.sqrt(len(fin_track)/len(all_tracksters_dec)))) - 1/2)
    else:
        Niter = 1
    if Printing:
        print("Total number of trackster in input: ", len(all_tracksters_dec))
        print("Number of iterations: ", Niter)
    
    #Grover loop:
    tmp_state = all_tracksters_qubit
    for i in range(int(Niter)):
        tmp_state = bb.dot(tmp_state)
        tmp_state = st_proj_op.dot(tmp_state)
        # print(tmp_state)
    
    #Find the resulting state:
    res_state = qubit_to_dec(tmp_state, all_points_ordered, form = output_form)
    #print("distances of resulting state = ",f_dist_t(res_state, dataset, "dec"))
    # print(res_state)
    res_state = int(res_state[::-1],2)
    # print(dec_to_qubit(43, all_points_ordered, form = "dec"))
    return res_state



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