
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code executes parallel multiprocessing of the analog SNN training algorithm
for each of the three noisy scenarios described in the paper. Here, the used benchmark is XOR.
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


from XOR_without_noise import assessing_without_noise
from XOR_cst_sd_noise import assessing_cst_sd_noise
from XOR_analog_sd_noise import assessing_analog_sd_noise



def parallel_executing(itera):
    '''
        Function that aggregates the functions intended for parallel execution
    '''
    if itera%3==0:
        assessing_cst_sd_noise(itera,dir_name='cst_noise_backup',noise=True)
    elif itera%3==1:
        assessing_analog_sd_noise(itera,dir_name='analog_noise_backup',noise=True)
    elif itera%3==2:
        assessing_without_noise(itera,dir_name='without_noise_backup',noise=False)

    return None


if __name__ == '__main__':
    '''
        This code use 'mutliprocessing library' to fully exploit the capabilities
        of the computer.
    '''

    amount_assessing = 150
    itera = [i for i in range(amount_assessing)]

    with multiprocessing.Pool(processes=30) as pool:
        nb_epoch_mean = pool.map(parallel_executing,itera)


