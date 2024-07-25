
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-03-23
# Source: https://github.com/YannaelB 
# Description: 
'''
This code execute a large amount of eSNN training in order to define the best best architecture.
This is a simple code running different proposed architectures and saving the number of epoch required
to solve the XOR problem. A genetic algorithm could be employed for better results at a higher cost.

It employs:
- 'eSNN_no_hidden_layer.py': eSNN with 2 input neurons and 13 output neurons. They are fully connected with 
excitatory and inhibitory synapses. There is a lateral inhibition in the output layer. The eNeuron model used
have a constant resistance and no inherent noise. This is a very simple model for fast simulations.
- 'eSNN_inherent_noise.py': Same architecture as 'eSNN_no_hidden_layer.py' but with inherent noise in the eNeuron model.
- 'eSNN_no_inhi.py': Same architecture as 'eSNN_no_hidden_layer.py' but without the inhibitory synapses between input and output.
- 'eSNN_without_wta.py': Same architecture as 'eSNN_no_hidden_layer.py' but without the lateral inhibition.
- 'eSNN_neuron_model_v3.py': Same architecture as 'eSNN_no_hidden_layer.py' but with a variable resistance in the eNeuron model.
- 'eSNN_1_hidden_layer.py': eSNN with 2 input neurons, 1 hidden layer with 7 neurons, 6 output neurons. Fully 
connected excitatory synapses between input and hidden layers, and between hidden and output layers. There is a
lateral inhibition in the output layer.
''' 
# --------------------------------------------------------------

# used libraries
import csv
import re
import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import random
import multiprocessing


from eSNN_no_hidden_layer import assessing_no_hidden_layer
from eSNN_inherent_noise import assessing_inherent_noise
from eSNN_no_inhi_connect import assessing_no_inhi_connect
from eSNN_without_wta import assessing_without_wta
from eSNN_neuron_model_v3 import assessing_neuron_model_v3
from eSNN_1_hidden_layer import assessing_1_hidden_layer



def parallel_executing(itera):
    '''
        Function that aggregates the functions intended for parallel execution
    '''
    if itera%6==0:
        assessing_no_hidden_layer(itera)
    if itera%6==1:
        assessing_inherent_noise(itera)
    if itera%6==2:
        assessing_no_inhi_connect(itera)
    if itera%6==3:
        assessing_without_wta(itera)
    if itera%6==4:
        assessing_neuron_model_v3(itera)
    if itera%6==5:
        assessing_1_hidden_layer(itera)
    return None


if __name__ == '__main__':
    '''
        This code use 'mutliprocessing library' to fully exploit the capabilities
        of the computer.
    '''

    amount_assessing = 100
    itera = [i for i in range(amount_assessing)]

    print("STARTING multiprocessing")
    with multiprocessing.Pool(processes=8) as pool:
        nb_epoch_mean = pool.map(parallel_executing,itera)


    print("main done !")
