
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code provides a collection of utility functions to support the main functionality of the XOR_'scenario-k'_noise.py scripts.
The functions in this module perform various tasks such as data processing, file handling, and computation.

Functions available:
- batch_data: Creates input data for the XOR problem.
- from_bit_to_current: Converts binary input into input current.
- from_bit_to_label_2: Converts binary input into binary scalar output (XOR).
- from_bit_to_label_4: Converts binary input into scalar output.
- from_label_to_XOR: Converts scalar input into binary XOR output.
- weight_evolution: Displays a graph of the evolution of synaptic weights conducted by STDP rules.
- weight_evolution_png: Displays a graph of the evolution of synaptic weights and saves it as PNG files.
- plot_accuracy_evol: Displays a graph of the evolution of accuracy.
- input_3D_vec: Creates input data for 3D graphs of the prediction from the SNN.
- reducing_list: Reduces the size of a list to lessen computation.
- extracting_eNeuron_behavior: Extracts data from the eNeuron transfer function and defines a linear interpolation function.
- save_accuracy_txtfile: Saves accuracy results to a text file.
- csv_save: Saves data to a CSV file.

# --------------------------------------------------------------
'''


#Used libraries
import csv
import re
import os
from brian2 import *
import numpy as np
from scipy.interpolate import interp1d
import brian2.numpy_ as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
import time
import warnings
import random
from random import shuffle
import pandas as pd
import pickle


def batch_data(n_train,n_test):
    '''
        Function that creates input data for the XOR problem
    In input:
        - n_train is the number of binary input [[0,0],[0,1],[1,0],[1,1]] for training the SNN to XOR problem
        - n_test is the number of binary input [[0,0],[0,1],[1,0],[1,1]] for testing the SNN to XOR problem
    It returns:
        - 2 vectors of length (n_train*4,2)and (n_train*4,1) (features and labels)
        - 2 vectors of length (n_test*4,2) and (n_test*4,1) (features and labels)
    '''
    X_train = n_train*[[0,0]] + n_train*[[0,1]] + n_train*[[1,0]] + n_train*[[1,1]]
    shuffle(X_train)
    y_train = [a ^ b for a, b in X_train]

    X_test = n_test*[[0,0]] + n_test*[[0,1]] + n_test*[[1,0]] + n_test*[[1,1]]
    shuffle(X_test)
    y_test = [a ^ b for a, b in X_test]
    
    return X_train,y_train,X_test,y_test

def from_bit_to_current(x):
    '''
        Function that converts binary into input current
    In input:
        - binary input (list)
    It returns:
        - binary input converting into input current (list without unit (/nA))
    '''
    if x == [0,0]:
        return [0.3,0.3]
    elif x == [0,1]:
        return [0.3,2]
    elif x == [1,0]:
        return [2,0.3]
    elif x == [1,1]:
        return [2,2]
    else:
        print(" WARNING PROBLEM !")
        
def from_bit_to_label_2(x):
    '''
        Function that converts binary input into binary output. It is helpful for saving SNN behavior
    In input:
        - binary input (list)
    It returns:
        - Xor operation of the input (scalar)
    '''
    if x == [0,0]:
        return 0
    elif x == [0,1]:
        return 1
    elif x == [1,0]:
        return 1
    elif x == [1,1]:
        return 0
    else:
        print(" WARNING PROBLEM !")
        
def from_bit_to_label_4(x):
    '''
        Function that converts binary input into scalar output. It is helpful for saving SNN behavior
    In input:
        - binary input (list)
    It returns:
        - scalar
    '''
    if x == [0,0]:
        return 0
    elif x == [0,1]:
        return 1
    elif x == [1,0]:
        return 2
    elif x == [1,1]:
        return 3
    else:
        print(" WARNING PROBLEM !")
        
def from_label_to_XOR(x):
    '''
        Function that converts input scalar into binary XOR operation. It is helpful for saving SNN behavior
    In input:
        - (scalar)
    It returns:
        - binary XOR (scalar)
    '''
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x == 2:
        return 1
    elif x == 3:
        return 0
    else:
        print(" WARNING PROBLEM !")


def weight_evolution(directory,itera,weight_evol_exci,weight_evol_inhi):
    ''''
        Function that display a graph of the evolution of the synaptic weights conducted by STDP rules 
    In input:
        - Directory where save the created image
        - 2 lists of the evolution of the synaptic weights conducted by STDP rules 
    It returns:
        - Nothing, it plots and saves the graph
    '''
    wmax=1
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1, 2, 1)  
    for idx_w in range(len(weight_evol_exci[0])):
        weight = [ligne[idx_w] for ligne in weight_evol_exci]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')
    plt.title('Evolution des poids synaptiques EXCI ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.ylim(-wmax-1, wmax+1) 
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for idx_w in range(len(weight_evol_inhi[0])):
        weight = [ligne[idx_w] for ligne in weight_evol_inhi]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')  
    plt.title('Evolution des poids synaptiques INHI ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"data_backup_test/{directory}/{directory+str(itera)}/weight_evolution_epoch.png")
    plt.close()


def weight_evolution_png(directory,itera,weight_evol_exci,weight_evol_inhi,labeled_neurons):
    ''''
        Function that display a graph of the evolution of the synaptic weights conducted by STDP rules 
    In input:
        - 2 lists of the evolution of the synaptic weights conducted by STDP rules 
    It returns:
        - Nothing, it plots and saves the graph
    '''
    for k in range(len(weight_evol_exci)):
        plt.figure()
        weight_visu = np.zeros((5,13))
        weight_visu[:2,:] = np.reshape(weight_evol_exci[k], (2, 13))
        weight_visu[2:4,:] = np.reshape(weight_evol_inhi[k], (2, 13))
        weight_visu[4,:] = labeled_neurons

        plt.imshow(weight_visu, cmap='gray', interpolation='nearest')
        plt.savefig(f"data_backup_test/{directory}/{directory+str(itera)}/weight_evolution_epch_{10*k}.png")
        plt.close()
        


def plot_accuracy_evol(directory,itera,accuracy_evol):
    ''''
        Function that display a graph of the average evolution of the accuracies (3 used criterias)
    In input:
        - list of the accuracies evolution
    It returns:
        - Nothing, it saves and plots the graph
    '''
    plt.figure(figsize=(10, 6)) 
    plt.plot(range(len(accuracy_evol)),accuracy_evol,label="max criteria")
    plt.title(' accuracy evolution ')
    plt.xlabel(f' epoch  ')
    plt.ylabel(' accuracy ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"data_backup_test/{directory}/{directory+str(itera)}/accuracy_evol_epoch.png")


def input_3D_vec(n_point = 5):
    '''
        Function that creates this axis and input for the 3D graphs of the prediction from the SNN (post-trained-XOR problem)
    In input:
        - n_point: number of point on 1 axis of the 3D graph
    It returns:
        - X_3D represents x-axis on the 3D graph (size list (n_point**2,1))
        - Y_3D represents x-axis on the 3D graph (size list (n_point**2,1))
        - input_3D_current is the input list for the SNN (size list (n_point**2,2))
    '''
    binary_list = [i / (n_point - 1) for i in range(n_point)]

    input_3D = np.zeros((2,len(binary_list)**2))
    X_3D,Y_3D = [],[]
    input_3D_current = []

    k = 0
    for i,binary_x in enumerate(binary_list):
     
        for j,binary_y in enumerate(binary_list):
            input_3D[0,k], input_3D[1,k] = binary_x,binary_y
            X_3D.append(binary_x)
            Y_3D.append(binary_y)
            input_3D_current.append([0.3+1.7*binary_x,0.3+1.7*binary_y])
            k+=1

    return X_3D,Y_3D,input_3D_current


def reducing_list(original_list, new_size):
    '''
        Function that reduces the size of a list and lessen the amount of computation
    In input:
        - Original_list: list you want to reduce in size
        - new_size: size desired for the list
    It returns:
        - Similar list with desired size
    '''
    original_size = len(original_list)
    if new_size >= original_size:
        return original_list  

    # Calculate the sampling interval
    interval = original_size / new_size

    # Sample elements at regular intervals
    new_list = [original_list[int(i * interval)] for i in range(new_size)]

    return new_list

def extracting_eNeuron_behavior(csv_file_path,nb_point_interpolation):
    '''
        Function that extrates data from the transfer function of th eNeuron ML in order to define a linear-
        interpolation function
    In input:
        - path of the csv file containing the transfer function of the eNeuron
        - nb_point_interpolation: size of the interpolated list
    It returns:
        - i_new: x-axis of the interpolated function (amp)
        - f_new: y-axs of the interpolated function (Hz)
    '''

    input_current = []
    spike_rate = []

    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        next(csv_reader)
        for row in csv_reader:
            values = re.split(r',', row[0])
            input_current.append(float(values[0]))
            spike_rate.append(float(values[1]))

    prefs.codegen.target = 'numpy'
    i_new = reducing_list(input_current,nb_point_interpolation)
    f_new = reducing_list(spike_rate,nb_point_interpolation)

    # Create the lookup table for the input currents
    lookup_current = np.linspace(0, input_current[-1], 50000)
    # Precompute the interpolated spike rates for these currents
    precomputed_spike_rate = np.interp(lookup_current, input_current, spike_rate)

    # Convert data to use with Brian2
    i_new = i_new * amp
    f_new = f_new * Hz
    
    return i_new, f_new, lookup_current,precomputed_spike_rate


def save_accuracy_txtfile(filename, n_epoch,accuracy_max,accuracy):
    '''
        Function that saves accuracy results to a text file
    '''
    with open(filename, 'a') as file:
        file.write(f"\n Method unsupervised : \n")
        file.write(f" It has taken "+ str(n_epoch) +f" epoch for reaching {accuracy_max*100} '%' accuracy \n")
        file.write(" The three observed accuracy are are the end of the training = ")
        accuracy_str = np.array_str(accuracy)
        file.write(accuracy_str)
        file.write("\n")
        file.write("\n")


def csv_save(title,vecteur):
    '''
        Function that saves data to a CSV file. Those files can be processed with 'main_processing.py'
    '''
    with open(f'{title}', 'a', newline='') as fichier:
        writer = csv.writer(fichier)
        writer.writerow(vecteur)