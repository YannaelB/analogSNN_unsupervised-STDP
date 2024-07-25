
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-03-23
# Source: https://github.com/YannaelB 
# Description: 
'''
This code execute a large amount of eSNN training in order to define the best number of neurons in the output layer.
The two criteria are the smallest number of neuron and the fastest convergence to 100% accuracy.

It employs 'unsupervised_STDP.py'
'''
# --------------------------------------------------------------


import csv
import re
import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import random
from unsupervised_STDP import assessing_network


def save_accuracy_txtfile(filename, n_output, mean_epoch):
    with open(filename, 'a') as file:
        file.write(f"\n Method unsupervised with  {n_output} output neurons: \n")
        file.write(" It has taken on average = "+ str(mean_epoch) +" epoch for reaching 100'%' accuracy \n")
        file.write("\n")


import multiprocessing
if __name__ == '__main__':

    nb_output = [6,7,8,9,10,11,12,13,14,16]
    with multiprocessing.Pool(processes=28) as pool:
        nb_epoch_mean = pool.map(assessing_network,nb_output)
        save_accuracy_txtfile(f"Accuracy_nb_epoch_save.txt", nb_epoch_mean[0][1], nb_epoch_mean[0][0])


