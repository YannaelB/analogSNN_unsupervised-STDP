
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
The code below implements unsupervised STDP-based learning within an analog SNN
using complete model of a Morris-Lecar electronic neuron and conductance-based synapses.
The neuron model is derived from the post layout simulation results
of its analog circuit from Ferreira et al. (2019)

The training, labeling, and evaluation processes are inspired by the methodology of Diehl&Cook (2015).
To maintain effective unsupervised learning, training, labeling and testing are implemented as three distinct
functions. However a training function with continous labeling and testing is also feasible.

Here, the analog SNN is tested on the XOR problem. 
The analog SNN is trained with neurons presenting intrinsic random noise (SCENARIO 1 in the paper)

This code employs the architecture : [2,13]+exci+inhi+wta.
It means that there is 2 input neurons fully connected with 13 outputs neurons
with both excitatory and inhibitory synapses conducted with STDP rules. Plus, there is a lateral inhibition
playing a soft-Winner-Take-All synaptic connection between output neurons. It implies 208 synapses. 


It operates and returns:
    - Creates necessary directories for data storage.
    - Creates XOR dataset in batches.
    - Initializes and configures the analog SNN model.
    - Evaluates the initial performance of the model to establish a baseline.
    - Trains the analog SNN model over several epochs, evaluating and saving accuracy after each epoch.
    - Saves the synaptic weights and neuron labels after training.
    - Saves the accuracy evolution data and plots the accuracy evolution.
    - Performs a final evaluation of the trained networt.

Example usage:
    assessing_cst_sd_noise(0,'cst_noise',noise=True)

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
from tools_XOR import *


from brian2 import clear_cache
from brian2 import device
device.reinit()
start_scope()

# Simulation parameters
defaultclock.dt = 0.01*us  
time_per_sample = 100 * us
resting_time = 20 * us

nb_point_interpolation = 1101
nb_point = 50000

# Defining the linear interpolation of the eNeuron ML behavior
i_new, f_new, lookup_current,precomputed_spike_rate = extracting_eNeuron_behavior(csv_file_path='FT_augmented.csv',nb_point_interpolation=nb_point_interpolation)
interp_function = interp1d(i_new, f_new, kind='linear', fill_value='extrapolate',assume_sorted=True)

@check_units(x=amp, result = Hz)
def y_interp_function(x):
    return interp_function(x)*Hz


# This method has found no significant benefit in timing computation:
'''
# lookup_currents = lookup_current * amp
# precomputed_spike_rates = precomputed_spike_rate * Hz

# # Function to get the precomputed spike rate from the lookup table
# @check_units(x=amp, result=Hz)
# def y_interp_function(x):
#     idx = np.searchsorted(lookup_currents, x) - 1
#     #idx = np.clip(idx, 0, nb_point - 2) 
#     return precomputed_spike_rates[idx]
'''

#Model parameters
v_rest = -80*mV
v_threshold = 60*mV  
v_reset = -100*mV
v_std_thresh = 3.3*mV #standart deviation for noise in scenario 1. The random noise is added throught the threshold level 
v0 = -40*mV  
tau = 500*us
tau2 = 15*us
Rm = 5000*Mohm
I0=1*amp
max_input_current = i_new[-1]

a_v = 1126.6343973469338
b_v = 169200.45773494235
a_value = 6.23667974e13

x_threshold_v= 8.3e-11*amp
q, R_ds,C_m,scale = 1.602176634*1e-19*amp*second, 10*1e6*ohm, 9.83*1e-15*second/ohm,15
avoid_error = 0.01*pA
frequency_min = 0.1*Hz
refrac = 0*us
alpha_e = 0.15
alpha_i = 0.04
beta_e = 1.9
beta_i = 0.5
avoid_error = 0.05*pA


@check_units(x=amp, result = amp)
def max_function_I(x):
    y = [max(i,avoid_error) for i in x]
    return y


V_x, V_y, V_z= 8*mV,62*mV,90*mV
R_x = Rm*(V_x/mV)/(15/60)/(abs(v_threshold/mV)+abs(v_reset/mV))  #New resistance for rising till V_x
R_y = Rm*(V_y/mV)/(35/60)/(abs(v_threshold/mV)+abs(v_reset/mV)) #New resistance for rising till V_y+V_x
R_z = Rm*(V_z/mV)/(10/60)/(abs(v_threshold/mV)+abs(v_reset/mV)) #New resistance for rising till V_threshold
k_redres = 0.007704519252680564
k_redres = -0.0018141002616710589


# This model use a logarithm-fit approximation rather than interpolation funcion, for fitting the transfer function of the ML eNeuron
neuron_eqs_v3_fit_function = '''
dv/dt =  (v_rest - v) / tau2 + (( int(I_total > x_threshold_v) *((a_v + b_v * log((1-int(I_total > x_threshold_v))+int(I_total > x_threshold_v)*(I_total+avoid_error)/(x_threshold_v) ))*I0)/a_value)*Rm) / tau : volt (unless refractory)
I_total = clip(Ie + I_inhi + I, 0*nA, 15*nA) : amp
Ie =  alpha_e*ge * uS *(80*mV -v)           : amp
I_inhi =  alpha_i*gi * uS *(-120*mV -v)           : amp
I : amp
dge/dt = -ge/(beta_e*us)            : 1
dgi/dt = -gi/(beta_i*us)            : 1
Rm = (R_x)*int(v<=(v_reset+V_x)) + (R_y)*int(v<=(v_reset+V_x+V_y))*(1-int(v<(v_reset+V_x))) + (R_z)*int(v>(v_reset+V_x+V_y)) : ohm

threshold = v_threshold + int(noise_activity)*v_std_thresh*randn() : volt (constant over dt)
noise_activity : boolean (shared) # (un)Activate the noise
'''

# This model use an interpolation funcion, for fitting the transfer function of the ML eNeuron
neuron_eqs_v3_interp_function = '''
dv/dt = (v_rest - v) / tau2 + redress*((y_interp_function(I_total)*I0/(a_value*Hz))*Rm)/ tau : volt (unless refractory)
I_total = clip(Ie + I_inhi + I, 0*nA, 15*nA) : amp
Ie =  alpha_e*ge * uS *(80*mV -v)           : amp
I_inhi =  alpha_i*gi * uS *(-120*mV -v)           : amp
I : amp
redress = 1+ k_redres*I_total/max_input_current     : 1
dge/dt = -ge/(beta_e*us)            : 1
dgi/dt = -gi/(beta_i*us)            : 1
Rm = (R_x)*int(v<=(v_reset+V_x)) + (R_y)*int(v<=(v_reset+V_x+V_y))*(1-int(v<(v_reset+V_x))) + (R_z)*int(v>(v_reset+V_x+V_y)) : ohm

threshold = v_threshold + int(noise_activity)*v_std_thresh*randn() : volt (constant over dt)
noise_activity : boolean (shared) # (un)Activate the noise
'''

reset_eqs = '''
v = v_reset
'''



#STDP parameters
tau_stdp = 3.98116837
mag_stdp = 0.007070372048
mag_stdp = 0.00310372048
taupre = tau_stdp*us
taupost = taupre
wmax = 1
dApre = mag_stdp
dApost = -dApre * taupre / taupost * 1.05

stdp_eqs_exci = '''
    w : 1
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
'''
pre_eqs_exci='''
    ge_post += w
    Apre += dApre
    w = clip(w + 0.3*lr*Apost, 0, wmax)
'''
post_eqs_exci='''
    Apost += dApost
    w = clip(w*(1-0.0007*lr) + lr*Apre, 0, wmax)
'''

stdp_eqs_inhi = '''
    w : 1
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
'''
pre_eqs_inhi='''
    gi_post += w
    Apre += dApre
    w = clip(w + 0.3*lr*Apost, 0, wmax)
'''
post_eqs_inhi='''
    Apost += dApost
    w = clip(w*(1-0.0007*lr) + lr*Apre, 0, wmax)
'''

synapse_model = "w : 1"



n_input, n_output = 2,13

class SNN_eNeuron():
    def __init__(self, debug=False):
        '''
            Initialization including setting up neuron groups and synapses
        '''

        device.reinit()
        start_scope()

        seed = (os.getpid() + int(time.time() * 1e6)) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

        # Simulation parameters
        defaultclock.dt = 0.015*us  

        model = {}

        # Input layer
        model['input'] = NeuronGroup(N=n_input, model=neuron_eqs_v3_interp_function, threshold='v >= threshold', reset=reset_eqs, refractory='refrac', method='heun',name='input')
        model['input'].noise_activity = True

        # output layer
        model['output'] = NeuronGroup(N=n_output, model=neuron_eqs_v3_interp_function, threshold='v >= threshold', reset=reset_eqs, refractory='refrac', method='heun',name='output')
        model['output'].noise_activity = True

        # Full connection from input to about with excitatory synapses conducted by STDP
        model['input_synapse_exci'] = Synapses(model['input'], model['output'], model=stdp_eqs_exci, on_pre=pre_eqs_exci, on_post=post_eqs_exci,name='input_synapse_exci')
        model['input_synapse_exci'].connect(True)
        model['input_synapse_exci'].lr = 0.5
        model['input_synapse_exci'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_exci'].delay = 'rand()*0*us'

        # Full connection from input to about with inhibitory synapses conducted by STDP
        model['input_synapse_inhi'] = Synapses(model['input'], model['output'], model=stdp_eqs_inhi, on_pre=pre_eqs_inhi, on_post=post_eqs_inhi,name='input_synapse_inhi')
        model['input_synapse_inhi'].connect(True)
        model['input_synapse_inhi'].lr = 0.5
        model['input_synapse_inhi'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_inhi'].delay = 'rand()*0*us'

        # Lateral inhibition with synapses set up with random but constant weights
        model['wta_synapse'] = Synapses(model['output'], model['output'], model=synapse_model, on_pre='gi_post += w',name='wta_synapse')
        model['wta_synapse'].connect(condition='i != j')
        model['wta_synapse'].w = 'rand() * wmax * 1.3 + 0.5*wmax'
        model['wta_synapse'].delay = 'rand()*0*us'

        model['output_SP'] = SpikeMonitor(model['output'], record=False,name='output_SP')

        if (debug):
            model['input_SP'] = SpikeMonitor(model['input'], record=True,name='input_SP')
            model['output_SP'] = SpikeMonitor(model['output'], record=True,name='output_SP')

        print("You've created :",len(model['input_synapse_exci'])+len(model['input_synapse_inhi'])+len(model['wta_synapse']) , "synapses in your network")

        self.net = Network(model)
        
        self.weight_evol_exci = []
        self.weight_evol_inhi = []
        self.accuracy_evol = []

    def __getitem__(self,key):
        return self.net[key]

    def training(self,idx_epoch,X_train,y_train,noise=True,plot=False):
        '''
            Training function with synaptic normalization and weight observation
        '''


        # (un)Activate noise during training
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Activate STDP learning
        self.net['input_synapse_exci'].lr = 1
        self.net['input_synapse_inhi'].lr = 1

        for j, (sample, label) in enumerate(zip(X_train, y_train)):

            self.net['input'].I = from_bit_to_current(sample) * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the training simulation
            self.net.run(time_per_sample)

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest


            # Normalization according to the divisive enforcement rule from Goodhill&Barrow (1994) :
            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j] = self.net['input_synapse_exci'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 1.3/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_exci'].w = weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j]

            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j] = self.net['input_synapse_inhi'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 0.6/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_inhi'].w = weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j]

        # Observation weight evolution
        weight = [self.net['input_synapse_exci'].w[i] for i in range(len(self.net['input_synapse_exci'].w))]
        self.weight_evol_exci.append(weight)
        weight = [self.net['input_synapse_inhi'].w[i] for i in range(len(self.net['input_synapse_inhi'].w))]
        self.weight_evol_inhi.append(weight)


    def labelisation(self,idx_epoch,X_train,y_train,noise=True,plot=False):
        '''
            Function for labeling output neurons based on their response to inputs
        '''

        # (un)Activate noise during training
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Unactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes2 = np.zeros((2, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        label_percent = 0.7 #30% of the previous trained data will be re-used to label output neurons
        nb_label = int(label_percent*len(y_train))
        X_labeling = X_train[nb_label:]
        y_labeling = y_train[nb_label:]


        for j, (sample, label) in enumerate(zip(X_labeling, y_labeling)):

            self.net['input'].I = from_bit_to_current(sample) * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes2[from_bit_to_label_2(sample)] += counter_spike    

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            old_spike_counts = np.copy(self.net['output_SP'].count)

        
        labeled_neurons2 = np.argmax(spikes2, axis=0)
        redundancy_neuron2 = np.bincount(labeled_neurons2, minlength=2)
        self.labeled_neurons2 = labeled_neurons2
        self.redundancy_neuron2 = redundancy_neuron2

        
    def evaluation(self,idx_epoch,X_test,y_test,noise=True,plot=False):
        '''
            Evaluation function that tests the trained network on test data and calculate accuracy
        '''
        # (un)Activate the noise
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Unactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes2 = np.zeros((2, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        num_correct_output_max2 = 0


        for j, (sample, label) in enumerate(zip(X_test, y_test)):
            self.net['input'].I = from_bit_to_current(sample) * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes2[from_bit_to_label_2(sample)] += counter_spike    
            
            # Prediction of the SNN post labelisation
            output_label_max2 = self.labeled_neurons2[np.argmax(counter_spike)]

            # Check prediction of the class of the sample
            if output_label_max2 == int(label):
                num_correct_output_max2 += 1
        

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            old_spike_counts = np.copy(self.net['output_SP'].count)


        accuracy_max = (num_correct_output_max2 / len(X_test))
        self.accuracy_evol.append(accuracy_max)

        # Export synaptic weights
        weight_wta = np.zeros((n_output*(n_output-1)))
        weight_wta = np.copy(self.net['wta_synapse'].w)

        weight_exci = np.zeros([n_input, n_output])
        weight_exci = np.copy(self.net['input_synapse_exci'].w)

        weight_inhi = np.zeros([n_input, n_output])
        weight_inhi = np.copy(self.net['input_synapse_inhi'].w)

        return accuracy_max, self.weight_evol_exci, self.weight_evol_inhi, self.labeled_neurons2,weight_exci,weight_inhi,weight_wta

    def plot_3D_XOR(self,idx_epoch,n_point=5,noise=True,plot=False):
        '''
            Function that plots 3D XOR problem representation after training
        '''
        # (un)Activate the noise
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        
        # Unactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0
        
        X_3D,Y_3D,input_3D_current = input_3D_vec(n_point)
        Z_3D = []

        spikes2 = np.zeros((2, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        for j, (sample) in enumerate(input_3D_current):

            self.net['input'].I = sample * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            
            # Predict the class of the sample
            output_label_max2 = self.labeled_neurons2[np.argmax(counter_spike)]
            Z_3D.append(output_label_max2)
            
            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest
            
            old_spike_counts = np.copy(self.net['output_SP'].count)

        if plot:
            ### PLOT 3D graph:

            # Given vectors
            x = np.array(X_3D)
            y = np.array(Y_3D)
            z = np.array(Z_3D)

            # Create a grid for the surface, this time ensuring x goes from 1 to 0
            xi = np.linspace(x.max(), x.min(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)

            # Linearly interpolate the z values on the grid
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')

            # Plotting the surface graph
            fig_surf = plt.figure()
            ax_surf = fig_surf.add_subplot(111, projection='3d')
            ax_surf.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none')
            ax_surf.set_xlim(1, 0)
            # Adding titles
            ax_surf.set_title('3D Surface Plot')
            ax_surf.set_xlabel('X')
            ax_surf.set_ylabel('Y')
            ax_surf.set_zlabel('Z')

            # Plotting the wireframe graph
            fig_wire = plt.figure()
            ax_wire = fig_wire.add_subplot(111, projection='3d')
            ax_wire.plot_wireframe(Xi, Yi, Zi, rstride=10, cstride=10)
            ax_wire.set_xlim(1, 0)

            # Adding titles
            ax_wire.set_title('3D Wireframe Plot')
            ax_wire.set_xlabel('X')
            ax_wire.set_ylabel('Y')
            ax_wire.set_zlabel('Z')

            plt.show()
        return X_3D,Y_3D,Z_3D



directory = 'directory_name'
itera = 0

def assessing_cst_sd_noise(iterat,dir_name,noise=True):
    '''
    Runs a simulation of an analog SNN on the XOR problem using eNeuron from scenario 1

    Parameters:
        iterat (int): An identifier to differentiate simulations when the function is called using the multiprocessing library.
        dir_name (str): Name of the directory where the data will be saved.
        noise (bool): Flag to activate intrinsic random noise in the neuron model.
    '''

    tic = time.time()

    
    global directory
    global itera
    directory = dir_name
    itera = iterat
    os.makedirs(f'data_backup_test/{directory}',exist_ok=True)
    os.makedirs(f'data_backup_test/{directory}/{directory+str(itera)}',exist_ok=True)

    accuracy_evol = []
    epoch_max = 100
    n_train,n_test = 10,7 #It means one epoch is composed of 10*[[0,0],[0,1],[1,0],[1,1]] (shuffled)
   
    # Creating the SNN model
    debug = False
    model = SNN_eNeuron(debug=debug)

    # Before initiating any training, the network should be tested to establish a baseline performance. 
    # The initial predictions are expected to be random, resulting in an accuracy of approximately 10%
    X_train,y_train,X_test,y_test = batch_data(n_train,n_test)
    model.labelisation(0,X_train,y_train,noise=True,plot=debug)
    accuracy_max,weight_evol_exci,weight_evol_inhi,labeled_neurons2,weight_exci,weight_inhi,weight_wta = model.evaluation(0,X_test,y_test,noise=True,plot=debug)
    accuracy_evol.append(accuracy_max)

    # Training
    for idx_epoch in range(epoch_max):

        X_train,y_train,X_test,y_test = batch_data(n_train,n_test)
        model.training(idx_epoch,X_train,y_train,noise=True,plot=debug)
        model.labelisation(idx_epoch,X_train,y_train,noise=True,plot=debug)

        accuracy_max,weight_evol_exci,weight_evol_inhi,labeled_neurons2,weight_exci,weight_inhi,weight_wta = model.evaluation(idx_epoch,X_test,y_test,noise=True,plot=debug)
        accuracy_evol.append(accuracy_max)

        if accuracy_max >= 0.90:
            # Stopping criterion is implemented because the learning rate for unsupervised STDP is not necessarily positive.
            break 

    ### Training finished, save the synaptic weights and the labeling to be able to rebuild the network:
    with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_exci', 'wb') as fp:
        pickle.dump(weight_exci, fp)
    with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_inhi', 'wb') as fp:
        pickle.dump(weight_inhi, fp)
    with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_wta', 'wb') as fp:
        pickle.dump(weight_wta, fp)
    with open(f'data_backup_test/{directory}/{directory+str(itera)}/labelfile', 'wb') as fp:
        pickle.dump(labeled_neurons2, fp)
    
    # Weight analysis
    weight_evolution(directory,itera,weight_evol_exci,weight_evol_inhi)
    csv_save(f'data_backup_test/{directory}/saving_accuracy_max.csv',accuracy_evol)
    csv_save(f'data_backup_test/{directory}/saving_nb_epoch_converging.csv',[idx_epoch+1])
    # Plot accuracy evolution
    plot_accuracy_evol(directory,itera,accuracy_evol)
    
    # Testing the trained network
    n_train,n_test = 25,25
    X_train,y_train,X_test,y_test = batch_data(n_train,n_test)
    accuracy_max,weight_evol_exci,weight_evol_inhi,labeled_neurons2,weight_exci,weight_inhi,weight_wta = model.evaluation(idx_epoch,X_test,y_test,noise=True,plot=debug)
    csv_save(f'data_backup_test/{directory}/saving_accuracy_final.csv',[accuracy_max])


    #3D figure
    X_3D,Y_3D,Z_3D = model.plot_3D_XOR(idx_epoch,n_point=10,noise=True,plot=False)
    csv_save(f'data_backup_test/{directory}/saving_Z_3D.csv',Z_3D)

    print(" DONE with assessing_cst_sd_noise with noise = ", noise, "in ",time.time()-tic, "s \n")

    return None


