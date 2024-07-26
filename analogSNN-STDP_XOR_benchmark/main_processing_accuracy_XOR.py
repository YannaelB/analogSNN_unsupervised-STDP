
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
This code processes data provided by 'main_training_XOR.py', saved in the directory 'data_backup'

The function performs the following tasks:
        1. Reads training and testing accuracy data from provided CSV file paths.
        2. Computes mean, standard deviation, and standard error of the mean for each training dataset (paths 1,2,3)
        3. Plots the accuracy evolution over training epochs for three scenarios.
        4. Displays final testing accuracy as boxplots for different scenarios (paths 11,12,13,21,31)
        5. Saves the plot as PDF and SVG files.

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
import matplotlib
import matplotlib.patches as mpatches


# Function that calculates the average number of epoch to achieve 100% accuracy to XOR problem
def average_nb_epoch(path):
    '''
        Function that calculates the average number of epoch to achieve 100% accuracy to XOR problem
    In input:
        - path of the directory containing the csv which backups the number of epoch to achieve 100% accuracy
    It returns:
        - The average number of epoch to achieve 100% accuracy
    '''
    df = pd.read_csv(f'{path}/saving_nb_epoch_converging.csv', header=None)
    data_nb_epoch= df.to_numpy()

    mean_nb_epoch = data_nb_epoch[0]
    for i in range(1,data_nb_epoch.shape[0]):
        mean_nb_epoch += data_nb_epoch[i]
    mean_nb_epoch = mean_nb_epoch/data_nb_epoch.shape[0]
    print(f"On average, it takes {mean_nb_epoch} for getting a 100'%' accuracy with {path} ")
    return mean_nb_epoch[0]


def csv_readable(path_csv):
    '''
        Function that reads CSV files of accuracy evolution backups:
        It makes a copy of the pointed file and had a 100-sized list at the position 0. 
        It makes pandas-reader-function considered the csv as a matrix nb_ligne*100 with NaN value to fill the gaps.
        It replaces 'NaN' values with the preceding no-NaN value. This implies that we stop 
        the training once an accuracy (other the 3 criterias) reaches 100%.
        Accuracies are considered constants beyond this point
    In input: 
        - path of the desired csv file, without the .csv
        - The average number of epoch to achieve 100% accuracy
    it returns:
        - An array of size [number of line of the pointed csv,1.5*mean_nb_epoch]
    '''
    epoch_max = 101 # Maximum of epoch during the training
    debug_line = [None for _ in range(epoch_max)]

    path_csv_copy = path_csv+'_processed.csv'
    shutil.copyfile(path_csv+'.csv', path_csv+'_processed.csv')

    # Make each line of the accuracy's csv having 'epoch_max' number of column
    with open(path_csv_copy, 'r', newline='') as file:
        x  = file.readlines()
    with open(path_csv_copy, 'w', newline='') as file:
        pen = csv.writer(file)
        pen.writerow(debug_line)
        for ligne in x:
            file.write(ligne)
    # Read the file-copy with pandas
    df = pd.read_csv(path_csv_copy, header=None)
    data_accuracy = df.to_numpy()
    
    for i in range(data_accuracy.shape[0]):
        for j in range(data_accuracy.shape[1]):
            if np.isnan(data_accuracy[i,j]):
                data_accuracy[i,j] = data_accuracy[i,j-1]

    reducing_limit = epoch_max-1
    data_accuracy_redu = data_accuracy[1:, :reducing_limit] # [1:,] is  is used to not count the previously added artificial line

    return data_accuracy_redu 




def accuracy_evolution(path1,path2,path3,path11,path12,path13,path21,path31,plot=False):
    '''
        Function that display a graph of the average evolution of the training and testing accuracy (3 used criterias)
    In input:
        - path1: Path to the CSV file containing training accuracy data for scenario 3.
        - path2: Path to the CSV file containing training accuracy data for scenario 1.
        - path3: Path to the CSV file containing training accuracy data for scenario 2.
        - path11: Path to the CSV file containing final testing accuracy data for the scenario 3.
        - path12: Path to the CSV file containing final testing accuracy data for the scenario 3.1 = noiseless + scenario 1.
        - path13: Path to the CSV file containing final testing accuracy data for the scenario 3.2 = noiseless + scenario 2.
        - path21: Path to the CSV file containing final testing accuracy data for scenario 1.
        - path31: Path to the CSV file containing final testing accuracy data for scenario 2.
        - plot: Boolean flag indicating whether to plot the graph.
    It returns:
        - None. This function does not return any values but plots and saves the graph if plot is set to True.
        - This function print the averages and medians
    '''
    # Training accuracy
    if path1:
        data_accuracy_max_redu1 = csv_readable(f'{path1}/saving_accuracy_max')

        # Calculate standard deviation
        std_dev1 = np.std(data_accuracy_max_redu1, axis=0)
        std_errs1 = sem(data_accuracy_max_redu1, axis=0)

        accuracy_max1 = np.copy(data_accuracy_max_redu1[0])
        n_1 = data_accuracy_max_redu1.shape[0]
        for i in range(1,data_accuracy_max_redu1.shape[0]):
            accuracy_max1 += np.copy(data_accuracy_max_redu1[i])
        # Calculate the average evolution
        accuracy_max1 = accuracy_max1/data_accuracy_max_redu1.shape[0]

    # Training accuracy
    if path2:
        data_accuracy_max_redu2 = csv_readable(f'{path2}/saving_accuracy_max')

        # Calculate standard deviation
        std_dev2 = np.std(data_accuracy_max_redu2, axis=0)
        std_errs2 = sem(data_accuracy_max_redu2, axis=0)

        accuracy_max2 = np.copy(data_accuracy_max_redu2[0])
        n_2 = data_accuracy_max_redu2.shape[0]
        for i in range(1,data_accuracy_max_redu2.shape[0]):
            accuracy_max2 += np.copy(data_accuracy_max_redu2[i])
        # Calculate the average evolution
        accuracy_max2 = accuracy_max2/data_accuracy_max_redu2.shape[0]

    # Training accuracy
    if path3:
        data_accuracy_max_redu3 = csv_readable(f'{path3}/saving_accuracy_max')

        # Calculate standard deviation
        std_dev3 = np.std(data_accuracy_max_redu3, axis=0)
        std_errs3 = sem(data_accuracy_max_redu3, axis=0)

        accuracy_max3 = np.copy(data_accuracy_max_redu3[0])
        n_3 = data_accuracy_max_redu3.shape[0]
        for i in range(1,data_accuracy_max_redu3.shape[0]):
            accuracy_max3 += np.copy(data_accuracy_max_redu3[i])
        # Calculate the average evolution
        accuracy_max3 = accuracy_max3/data_accuracy_max_redu3.shape[0]


    # Final Testing accuracy
    if path11: #Noiseless
        data_accuracy_11 = pd.read_csv(f'{path11}',header=None)
        mean11 = np.mean(data_accuracy_11)
        median11 = data_accuracy_11.median()
        print(" Scenario 3, mean = ",mean11, " and median = ",median11,"shape = ",data_accuracy_11.shape," \n")

    if path12:
        data_accuracy_12 = pd.read_csv(f'{path12}',header=None)
        print(" Scenario 3.1, mean = ",np.mean(data_accuracy_12), " and median = ",data_accuracy_12.median()," \n" )


    if path13: #noiseless + scenario 2
        data_accuracy_13 = pd.read_csv(f'{path13}',header=None)
        print(" Scenario 3.2, mean = ",np.mean(data_accuracy_13), " and median = ",data_accuracy_13.median()," \n" )


    if path21: #scenario 1
        data_accuracy_21= pd.read_csv(f'{path21}',header=None)
        print(" Scenario 1, mean = ",np.mean(data_accuracy_21), " and median = ",data_accuracy_21.median()," \n" )

    
    if path31: #scenario 2
        data_accuracy_31= pd.read_csv(f'{path31}',header=None)
        print(" Scenario 2, mean = ",np.mean(data_accuracy_31), " and median = ",data_accuracy_31.median()," \n" )
       

    
    if plot:
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        rc('text', usetex=True)

        plt.figure(figsize=(9, 7))

        if path11: #Boxplot
            boxplotElements = plt.boxplot(data_accuracy_11.values, positions=[len(accuracy_max1)+0.4], widths=2.6, patch_artist=True,showfliers=True)
            for element in boxplotElements['medians']:
                element.set_color('black')
                element.set_linewidth(1)
            for element in boxplotElements['boxes']:
                element.set_edgecolor('b')
                element.set_facecolor((117/255, 175/255, 244/255))
                element.set_linewidth(1)
                element.set_linestyle('-')
                element.set_fill(True)
            for element in boxplotElements['whiskers']:
                element.set_color('b')
                element.set_linewidth(1)
            for element in boxplotElements['caps']:
                element.set_color('b')
                element.set_linewidth(1)
            for element in boxplotElements['fliers']:
                element.set(marker='.', color='b', markerfacecolor='b', markeredgecolor='b', alpha=0.8)
            box_patch11 = mpatches.Patch(facecolor=(117/255, 175/255, 244/255), edgecolor='b', linewidth=2, linestyle='-',label='scenario 3')


        if path12: #Boxplot
            boxplotElements = plt.boxplot(data_accuracy_12.values, positions=[len(accuracy_max1)+4], widths=2.6, patch_artist=True,showfliers=True)
            for element in boxplotElements['medians']:
                element.set_color('black')
                element.set_linewidth(1)
            for element in boxplotElements['boxes']:
                element.set_edgecolor('#C4E302')
                element.set_facecolor('#E0EE89')
                element.set_linewidth(1)
                element.set_linestyle('-')
                element.set_fill(True)
            for element in boxplotElements['whiskers']:
                element.set_color('#C4E302')
                element.set_linewidth(1)
            for element in boxplotElements['caps']:
                element.set_color('#C4E302')
                element.set_linewidth(1)
            for element in boxplotElements['fliers']:
                element.set(marker='.', color='#C4E302', markerfacecolor='#C4E302', markeredgecolor='#C4E302', alpha=0.8)
            box_patch12 = mpatches.Patch(facecolor='#E0EE89', edgecolor='#C4E302', linewidth=2, linestyle='-',label='scenario 3.1')


        if path13: #Boxplot
            boxplotElements = plt.boxplot(data_accuracy_13.values, positions=[len(accuracy_max1)+8], widths=2.6, patch_artist=True,showfliers=True)
            for element in boxplotElements['medians']:
                element.set_color('black')
                element.set_linewidth(1)
            for element in boxplotElements['boxes']:
                element.set_edgecolor('#F311E2')
                element.set_facecolor('#EA83E2')
                element.set_linewidth(1)
                element.set_linestyle('-')
                element.set_fill(True)
            for element in boxplotElements['whiskers']:
                element.set_color('#F311E2')
                element.set_linewidth(1)
            for element in boxplotElements['caps']:
                element.set_color('#F311E2')
                element.set_linewidth(1)
            for element in boxplotElements['fliers']:
                element.set(marker='.', color='#F311E2', markerfacecolor='#F311E2', markeredgecolor='#F311E2', alpha=0.8)
            box_patch13 = mpatches.Patch(facecolor='#EA83E2', edgecolor='#F311E2', linewidth=2, linestyle='-',label='scenario 3.2')


        if path21: #Boxplot
            boxplotElements = plt.boxplot(data_accuracy_21.values, positions=[len(accuracy_max1)+12], widths=2.6, patch_artist=True,showfliers=True)
            for element in boxplotElements['medians']:
                element.set_color('black')
                element.set_linewidth(1)
            for element in boxplotElements['boxes']:
                element.set_edgecolor('green')
                element.set_facecolor((0.6, 0.99, 0.6))
                element.set_linewidth(1)
                element.set_linestyle('-')
                element.set_fill(True)
            for element in boxplotElements['whiskers']:
                element.set_color('green')
                element.set_linewidth(1)
            for element in boxplotElements['caps']:
                element.set_color('green')
                element.set_linewidth(1)
            for element in boxplotElements['fliers']:
                element.set(marker='.', color='green', markerfacecolor='green', markeredgecolor='green', alpha=0.8)
            box_patch21 = mpatches.Patch(facecolor=(0.6, 0.99, 0.6), edgecolor='green', linewidth=2, linestyle='-',label='scenario 1')

        if path31: #Boxplot
            boxplotElements = plt.boxplot(data_accuracy_31.values, positions=[len(accuracy_max1)+16], widths=2.6, patch_artist=True,showfliers=True)
            for element in boxplotElements['medians']:
                element.set_color('black')
                element.set_linewidth(1)
            for element in boxplotElements['boxes']:
                element.set_edgecolor('red')
                element.set_facecolor((0.98, 0.5, 0.447))
                element.set_linewidth(1)
                element.set_linestyle('-')
                element.set_fill(True)
            for element in boxplotElements['whiskers']:
                element.set_color('red')
                element.set_linewidth(1)
            for element in boxplotElements['caps']:
                element.set_color('red')
                element.set_linewidth(1)
            for element in boxplotElements['fliers']:
                element.set(marker='.', color='red', markerfacecolor='red', markeredgecolor='red', alpha=0.8)
            box_patch31 = mpatches.Patch(facecolor=(0.98, 0.5, 0.447), edgecolor='red', linewidth=2.6, linestyle='-',label='scenario 2')

        # Vertical line to draw a boundary between trainning and testing
        plt.axvline(x=len(accuracy_max1)-1, color='black', linestyle='-',linewidth=0.55)
        if path2:
            plt.plot(range(len(accuracy_max2)),accuracy_max2,"g",label=f"scenario 1")
            plt.fill_between(range(len(accuracy_max2)), accuracy_max2 - std_errs2, accuracy_max2 + std_errs2, color=(0.6, 0.99, 0.6), alpha=0.4)
        if path3:
            plt.plot(range(len(accuracy_max3)),accuracy_max3 ,"r",label=f"scenario 2")
            plt.fill_between(range(len(accuracy_max3)), accuracy_max3 - std_errs3, accuracy_max3 + std_errs3, color=(0.98, 0.5, 0.447), alpha=0.4)
        if path1:
            plt.plot(range(len(accuracy_max1)),accuracy_max1,"b",label=f"scenario 3")
            plt.fill_between(range(len(accuracy_max1)), accuracy_max1 - std_errs1, accuracy_max1 + std_errs1, color=(117/255, 175/255, 244/255), alpha=0.4)
        
        plt.xlabel(f' Epoch ',fontsize=23)
        plt.xticks(np.linspace(0,99,11),range(0,110,10),fontsize=22)
        plt.ylabel(' Training accuracy [\%]',fontsize=23)
        plt.yticks(np.linspace(0.2,1,9),['20','30','40','50','60','70','80','90','100'],fontsize=22)

        ax1 = plt.gca()  # Get the current axis
        ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis
        ax2.set_yticks(ax1.get_yticks())  
        ax2.set_yticklabels(ax1.get_yticklabels(), fontsize=22)
        ax2.set_ylabel('Post training accuracy [\%]', fontsize=23)
        ax1.set_ylim(0.1, 1.03)
        ax2.set_ylim(0.1, 1.03)
        ax1.set_xlim(-1.5, 120)
        ax1.grid(True, which='both', axis='both')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.extend([box_patch21,box_patch31,box_patch11,box_patch12,box_patch13])
        labels.extend([box_patch21.get_label(),box_patch31.get_label(),box_patch11.get_label(),box_patch12.get_label(),box_patch13.get_label()])
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
        plt.legend(handles=handles, labels=labels, fontsize=20,loc='lower center')
        #plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/accuracy_evol_xor.pdf', format='pdf', bbox_inches='tight',dpi=300)
        plt.savefig(f"results/accuracy_evol_xor.svg")
        plt.show()



if __name__ == '__main__':

    # Path of the folder containing the data from the SNN you want to analyse
    directo = 'data_backup'

    # Accuracy_training
    path1 = f'{directo}/without_noise_backup' #scenario 3
    path2 = f'{directo}/cst_noise_backup' #scenario 1
    path3 = f'{directo}/analog_noise_backup' #scenario 2
    # Accuracy testing
    path11 = f'{directo}/without_noise_backup/saving_accuracy_final.csv' #scenario 3
    path12 = f'{directo}/without_noise_backup/saving_accuracy_cst_sd_noise.csv' #scenario 3.1 = noiseless + scenario 1
    path13 = f'{directo}/without_noise_backup/saving_accuracy_analog_sd_noise.csv' #scenario 3.2 = noiseless + scenario 2
    path21 = f'{directo}/cst_noise_backup/saving_accuracy_final.csv' #scenario 1
    path31 = f'{directo}/analog_noise_backup/saving_accuracy_final.csv' #scenario 2



    if path1:
        mean_nb_epoch = average_nb_epoch(path1)
        print(" mean_nb_epoch = ", mean_nb_epoch)
    if path2:
        mean_nb_epoch = average_nb_epoch(path2)
        print(" mean_nb_epoch = ", mean_nb_epoch)
    if path3:
        mean_nb_epoch = average_nb_epoch(path3)
        print(" mean_nb_epoch = ", mean_nb_epoch)

    accuracy_evolution(path1,path2,path3,path11,path12,path13,path21,path31, plot=True)
