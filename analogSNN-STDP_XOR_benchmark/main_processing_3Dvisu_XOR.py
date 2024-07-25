
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code processes data provided by 'main_training_XOR.py', saved in the directory 'data_backup'

The function performs the following tasks:
        1. Reads testing accuracy data from provided CSV file paths 'saving_Z_3D'
        2. Plots the averaged 3D graph representing the solvation of the XOR problem for each scenario (resulting in 5 graphs)
        5. Saves the plot as SVG files.

'''
# --------------------------------------------------------------

#Used libraries
import pandas as pd
import numpy as np
import csv
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import griddata
from scipy.stats import sem
from matplotlib import rc
from tools_XOR import *
import matplotlib
import matplotlib.patches as mpatches


def plot_3D_average(path,save_file,plot=False,axis=False):
    '''
        Function that display 3D graphs of the averaged prediction from the SNN (post-trained-XOR problem)
    In input:
        - path of the directory containing the csv which backups the prediction made by the SNN to [X,Y]-inputs
    It returns:
        - Nothing, it plots and saves the 3D graphs if plot==True
    '''

    df = pd.read_csv(f'{path}', header=None)
    data_Z_3D= df.to_numpy()

    # Calculates the average of the prediction made by the SNN
    Z_3D = data_Z_3D[0]
    for i in range(1,data_Z_3D.shape[0]):
        Z_3D += data_Z_3D[i]
    Z_3D = Z_3D/data_Z_3D.shape[0]

    X_3D,Y_3D,input_3D_current = input_3D_vec(int(np.sqrt(data_Z_3D.shape[1])))

    ### PLOT 3D graph:
    x = np.array(X_3D)
    y = np.array(Y_3D)
    z = np.array(Z_3D)

    # Create a grid for the surface
    xi = np.linspace(x.max(), x.min(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    Xi, Yi = np.meshgrid(xi, yi)

    # Linearly interpolate the z values on the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    if plot:
        # Plotting the surface graph
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        rc('text', usetex=True)
        

        fig_surf = plt.figure(figsize=(6.2, 6)) 
        ax_surf = fig_surf.add_subplot(111, projection='3d')
        ax_surf.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none')
        ax_surf.set_xlim(0, 1)
        ax_surf.set_ylim(1, 0)

        if axis == True:
            #plt.rcParams['font.family'] = 'Times New Roman'
            #plt.rcParams['font.size'] = 22  # Set the font size you'd like to use
            #plt.rcParams['mathtext.fontset'] = 'stix' 
            ax_surf.tick_params(axis='both', which='major', labelsize=22)
            ax_surf.set_yticks(np.linspace(0,1,6),['0.0','0.2','0.4','0.6','0.8',''])
            fig_surf.tight_layout()

        if axis == False :
            ax_surf.set_xlabel('')
            ax_surf.set_ylabel('')
            ax_surf.set_zlabel('')

            ax_surf.set_xticks(np.linspace(0,1,10),[])
            ax_surf.set_yticks(np.linspace(0,1,10),[])
            ax_surf.set_zticks(np.linspace(0,1,10),[])
            fig_surf.tight_layout()

        fig_surf.savefig(f"results/XOR_3D_{save_file}.svg",bbox_inches='tight')
        fig_surf.show()




global directo

if __name__ == '__main__':

    # Path of the folder containing the data from the SNN you want to analyse
    directo = 'data_backup'

    path_folder1 = f'{directo}/without_noise_backup/saving_Z_3D_without_noise.csv'  #scenario 3
    path_folder2 = f'{directo}/cst_noise_backup/saving_Z_3D.csv'  #scenario 1
    path_folder3 = f'{directo}/analog_noise_backup/saving_Z_3D.csv'  #scenario 2
    path_folder4 = f'{directo}/without_noise_backup/saving_Z_3D_cst_noise.csv' #scenario 3.1 = noiseless + scenario 1
    path_folder5 = f'{directo}/without_noise_backup/saving_Z_3D_analog_noise.csv' #scenario 3.2 = noiseless + scenario 2


    if path_folder1:
        plot_3D_average(path_folder1,'scenario3',plot=True,axis=True)
    if path_folder2:
        plot_3D_average(path_folder2,'scenario1',plot=True,axis=False)
    if path_folder3:
        plot_3D_average(path_folder3,'scenario2',plot=True,axis=False)
    if path_folder4:
        plot_3D_average(path_folder4,'scenario31',plot=True,axis=False) 
    if path_folder5:
        plot_3D_average(path_folder5,'scenario32',plot=True,axis=False)