
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code executes parallel multiprocessing of the analog SNN training and testing algorithms
for each of the three noisy scenarios described in the associated research paper. Here, the MNIST dataset is used
as the benchmark.

For further explanation, refer to the individual function documentation.
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
import multiprocessing


from mnist_without_noise import main_mnist_without_noise
from mnist_cst_sd_noise import main_mnist_cst_sd_noise
from mnist_analog_sd_noise import main_mnist_analog_sd_noise




def parallel_executing(itera):
    '''
        Function that aggregates the functions intended for parallel execution
    '''
    if itera%3==0:
        main_mnist_cst_sd_noise(itera,dir_name='cst_noise_final',noise=True) 
    if itera%3==1:
        main_mnist_without_noise(itera,dir_name='without_noise_final',noise=False) 
    if itera%3==2:
        main_mnist_analog_sd_noise(itera,dir_name='analog_noise_final',noise=True) 
    
    return None


if __name__ == '__main__':
    '''
        This code use 'mutliprocessing library' to fully exploit the capabilities
        of the computer.
    '''

    amount_assessing = 30
    itera = [i for i in range(amount_assessing)]

    with multiprocessing.Pool(processes=30) as pool:
        results = pool.map(parallel_executing,itera)

