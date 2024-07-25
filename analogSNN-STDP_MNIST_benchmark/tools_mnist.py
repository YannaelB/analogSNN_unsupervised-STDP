
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code provides a collection of utility functions to support the main functionality of the mnist_'scenario-k'_noise.py scripts.
The functions in this module perform various tasks such as data processing, file handling, and computation.

Functions available:
- load_mnist: Loads and preprocesses the MNIST dataset.
- shuffle_and_batch_data: Shuffles and batches the MNIST data.
- reducing_list: Reduces the size of a list for less computation.
- extracting_eNeuron_behavior: Extracts and interpolates data from the eNeuron simulation results.
- save_accuracy_txtfile: Saves accuracy results to a text file with detailed information.
- save_accuracy_txtfile_short: Saves accuracy results to a text file in a short format.
- weight_evolution: Plots the evolution of synaptic weights over epochs.
- plot_accuracy_evol: Plots the evolution of accuracies over epochs.
- csv_save: Saves data to a CSV file for further processing.

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
import time
import warnings
from scipy import interpolate
from keras.datasets import mnist
import random
import pickle
from random import shuffle
import seaborn as sn
import pandas as pd
import pickle
import multiprocessing



def load_mnist(nb_train, nb_test, working_range=3, batch_size_train=256, batch_size_test=50):
    '''
        Function that load MNIST dataset. 
    In input:
        - nb_train and nb_test are the total size of dataset you want to use for your simulation. It's >= max(batch_size_train,batch_size_test)
        - batch_size_train and batch_size_test is the size of the batches respectively for training and evaluation
        - working_range is the magnitude conversion of pixel values (from range 255 to range 'working_range')
    It returns:
        - 4 lists of batch data (Input and labels for training + Input and labels for testing)
    '''
    # Import the MNIST Database
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # simplified classification (0 1 2 3 4)
    # X_train = X_train[(y_train == 1) | (y_train == 0) | (y_train == 3) | (y_train == 2) | (y_train == 4)]
    # y_train = y_train[(y_train == 1) | (y_train == 0) | (y_train == 3) | (y_train == 2) | (y_train == 4)]
    # X_test = X_test[(y_test == 1) | (y_test == 0) | (y_test == 3) | (y_test == 2) | (y_test == 4)]
    # y_test = y_test[(y_test == 1) | (y_test == 0) | (y_test == 3) | (y_test == 2) | (y_test == 4)]

    # Make pixel intensity 255 become working_range*nA
    X_train = X_train * (working_range/255)
    X_test = X_test * (working_range/255)

    if nb_train >= len(X_train) or nb_train <= 0:
        nb_train = len(X_train)
    if nb_test >= len(X_test) or nb_test <= 0:
        nb_test = len(X_test)

    #Reduce the mnist-list for debbuging part
    X_train, X_test = X_train[:nb_train], X_test[:nb_test]
    y_train, y_test = y_train[:nb_train], y_test[:nb_test]

    X_train_batches, y_train_batches,X_test_batches, y_test_batches, nb_batch = shuffle_and_batch_data(X_train, y_train, X_test, y_test,batch_size_train,batch_size_test)
    
    return X_train_batches, y_train_batches, X_test_batches, y_test_batches,nb_batch



def shuffle_and_batch_data(X_train, y_train, X_test, y_test, batch_size_train,batch_size_test):
    '''
        Function that creates MNIST batch data into the previous function load_mnist() 
    '''
    # Combine the features and labels to shuffle them together
    combined_train = list(zip(X_train, y_train))
    np.random.shuffle(combined_train)
    X_train[:], y_train[:] = zip(*combined_train)

    combined_test = list(zip(X_test, y_test))
    np.random.shuffle(combined_test)
    X_test[:], y_test[:] = zip(*combined_test)

    # Calculate the total number of batches
    n_batches_train = len(X_train) // batch_size_train
    if len(X_train) % batch_size_train != 0:
        n_batches_train += 1

    # Calculate the total number of batches
    n_batches_test = len(X_test) // batch_size_test
    if len(X_test) % batch_size_test != 0:
        n_batches_test += 1

    # Divide the data into batches
    X_train_batches = np.array_split(np.array(X_train), n_batches_train)
    y_train_batches = np.array_split(np.array(y_train), n_batches_train)
    X_test_batches = np.array_split(np.array(X_test), n_batches_test)
    y_test_batches = np.array_split(np.array(y_test), n_batches_test)

    return X_train_batches, y_train_batches, X_test_batches, y_test_batches,min(n_batches_train,n_batches_test)



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
        Function that extrates data from from the post layout simulation results of the ML eNeuron analog circuit.
        It provides the transfer function in order to define a linear-interpolation function.
    In input:
        - path of the csv file containing the post-layout simulation results of the ML eNeuron
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


def save_accuracy_txtfile(filename, itera, n_epoch,accuracy_max,accuracy):
    '''
        Function that saves accuracy results into a text-file
    '''
    with open(filename, 'a') as file:
        file.write(f"\n Method unsupervised for iteration  {itera} : \n")
        file.write(f" It has taken "+ str(n_epoch) +f" epoch for reaching {accuracy_max*100} '%' accuracy \n")
        file.write(" The three observed accuracy are are the end of the training = ")
        accuracy_str = np.array_str(accuracy)
        file.write(accuracy_str)
        file.write("\n")
        file.write("\n")

def save_accuracy_txtfile_short(filename,accuracy):
    '''
        Function that saves accuracy results into a text-file
    '''
    with open(filename, 'a') as file:
        accuracy_str = np.array_str(accuracy)
        file.write(accuracy_str)
        file.write("\n")
        
def weight_evolution(directory,weight_evol_exci,weight_evol_inhi):
    ''''
        Function that display a graph of the evolution of the synaptic weights conducted by STDP rules.
        It plots some random synaptic weights because it's not possible to vizualize all of them.
    In input:
        - 2 lists of the evolution of the synaptic weights conducted by STDP rules 
    It returns:
        - save the figure of weight evolution depending on epoch increasing.
    '''
    random_idx_weight = [random.randint(0, len(weight_evol_exci[0])-1) for _ in range(50)]
    
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1, 2, 1)  
    for idx_w in random_idx_weight:
        weight = [ligne[idx_w] for ligne in weight_evol_exci]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')  
    plt.title('Excitatory weights ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for idx_w in random_idx_weight:
        weight = [ligne[idx_w] for ligne in weight_evol_inhi]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')  
    plt.title('Inhibitory weights ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.grid(True)
    
    plt.savefig(f"data_backup_test/{directory}/weight_evolution_epoch.png")
    
    
def plot_accuracy_evol(directory,accuracy_evol):
    ''''
        Function that display a graph of the average evolution of the accuracies 
    In input:
        - list of the accuracies evolution
    It returns:
        - Nothing, it saves and plots the graph if plot==True
    '''
    plt.figure(figsize=(10, 6)) 
    plt.plot(range(len(accuracy_evol)),accuracy_evol,label="most-spiking criterion")
    plt.title(' accuracy evolution ')
    plt.xlabel(f' epoch  ')
    plt.ylabel(' accuracy ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"data_backup_test/{directory}/accuracy_evol_epoch.png")


def csv_save(title,vector):
    '''
        Function that saves data into a CSV file. Those files can be processed with 'main_processing.py'
    '''
    with open(f'{title}', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(vector)


