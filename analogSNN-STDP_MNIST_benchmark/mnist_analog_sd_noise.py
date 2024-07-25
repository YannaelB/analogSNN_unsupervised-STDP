
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-05-29
# Source: https://github.com/YannaelB 
# Description: 
'''
The code below implements unsupervised STDP-based learning within a analog SNN
using complete model of a Morris-Lecar electronic neuron and conductance-based synapses.
The neuron model is derived from the post layout simulation results
of its analog circuit from Ferreira et al. (2019)

The training, labeling, and evaluation processes are inspired by the methodology of Diehl&Cook (2015).
To maintain effective unsupervised learning, training, labeling and testing are implemented as three distinct
functions. However a training function with continous labeling and testing is also feasible.

Here, the analog SNN is tested on the MNIST benchmark dataset. 
The analog SNN is trained with neurons presenting intrinsic random noise (SCENARIO 2 in the paper)

This code employs the architecture : [784,1225]+exci+inhi+wta.
It means that there is 784 input neurons fully connected with 1225 outputs neurons
with both excitatory and inhibitory synapses conducted with STDP rules. Plus, there is a lateral inhibition
playing a soft-Winner-Take-All synaptic connection between output neurons. It implies 3420200 synapses. 


It operates and returns:
    - Creates necessary directories for data storage.
    - Loads the MNIST dataset in batches.
    - Initializes and configures the analog SNN model.
    - Evaluates the initial performance of the model to establish a baseline.
    - Trains the SNN model over several epochs, evaluating and saving accuracy after each epoch.
    - Saves the synaptic weights and neuron labels after training.
    - Saves the accuracy evolution data and plots the accuracy evolution.
    - Performs a final evaluation of the trained network on the MNIST test set.
    - Generates and saves a confusion matrix for the final evaluation.

Example usage:
    acc = main_mnist_analog_sd_noise(0,'analog_noise_final',noise=True)


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
from tools_mnist import *


from brian2 import clear_cache
from brian2 import device
device.reinit()
start_scope()

# Simulation parameters
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
v_std_thresh = 3.3*mV #standart deviation for noise in scenario 1. The inherent noise is added throught the threshold level 
v_reset = -100*mV
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
q, R_ds,C_m,scale = 1.6*1e-19*amp*second, 5*ohm, 5*1e-8*second/ohm,15
avoid_error = 0.01*pA
frequency_min = 0.1*Hz
refrac = 0*us
alpha_e = 0.15
alpha_i = 0.04
beta_e = 1.9
beta_i = 0.5
avoid_error = 0.01*pA

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


# This model use an interpolation funcion, for fitting the transfer function of the ML eNeuron
neuron_eqs_v3_interp_function = '''
dv/dt = (v_rest - v) / tau2 + redress*((I_syn*I0/(a_value*Hz))*Rm)/ tau : volt (unless refractory)
I_total = clip(Ie + I_inhi + I, 0*nA, 15*nA) : amp
I_syn = y_interp_function(I_total) : Hz (constant over dt)
Ie =  alpha_e*ge * uS *(80*mV -v)           : amp
I_inhi =  alpha_i*gi * uS *(-120*mV -v)           : amp
I : amp
redress = 1+ k_redres*I_total/max_input_current     : 1
dge/dt = -ge/(beta_e*us)            : 1
dgi/dt = -gi/(beta_i*us)            : 1
Rm = (R_x)*int(v<=(v_reset+V_x)) + (R_y)*int(v<=(v_reset+V_x+V_y))*(1-int(v<(v_reset+V_x))) + (R_z)*int(v>(v_reset+V_x+V_y)) : ohm

threshold = v_threshold + int(noise_activity)*sigma*randn()*(v_threshold-v_reset)*frequency*sqrt(idx_spk) : volt (constant over dt)
frequency = I_syn : Hz (constant over dt)
sigma = int(frequency>frequency_min)*sqrt(scale*q/(I_total+avoid_error)*(1/(frequency+frequency_min) + R_ds*C_m/pi)) : second (constant over dt) #standart deviation
idx_spk : 1
noise_activity : boolean (shared) # (un)Activate the noise
'''


# This model use a logarithm-fit approximation rather than interpolation funcion, for fitting the transfer function of the ML eNeuron
neuron_eqs_v3_fit_function = '''
dv/dt =  (v_rest - v) / tau2 + ( int(I_total > x_threshold_v) *(I_syn*I0)/(a_value*Hz)*Rm) / tau : volt (unless refractory)
I_total = clip(Ie + I_inhi + I, 0*nA, 15*nA) : amp
I_syn = Hz*(a_v + b_v * log((1-int(I_total > x_threshold_v))+int(I_total > x_threshold_v)*(I_total+avoid_error)/(x_threshold_v) )) : Hz (constant over dt)
Ie =  alpha_e*ge * uS *(80*mV -v)           : amp
I_inhi =  alpha_i*gi * uS *(-120*mV -v)           : amp
I : amp
dge/dt = -ge/(beta_e*us)            : 1
dgi/dt = -gi/(beta_i*us)            : 1
Rm = (R_x)*int(v<=(v_reset+V_x)) + (R_y)*int(v<=(v_reset+V_x+V_y))*(1-int(v<(v_reset+V_x))) + (R_z)*int(v>(v_reset+V_x+V_y)) : ohm

threshold = v_threshold + int(noise_activity)*sigma*randn()*(v_threshold-v_reset)*I_syn*sqrt(idx_spk) : volt (constant over dt)
sigma = int(I_syn>frequency_min)*sqrt(scale*q/(I_total+avoid_error)*(1/(I_syn+frequency_min) + R_ds*C_m/pi)) : second (constant over dt) #standart deviation
idx_spk : 1
noise_activity : boolean (shared) # (un)Activate the noise
'''

reset_eqs = '''
v = v_reset
idx_spk = int(I_total>=7*pA)*(idx_spk) + 1 # reset the index spike occurrence if the current is below the resonance threshold
'''



#STDP parameters
tau_stdp = 3.98116837
mag_stdp = 0.007070372048
mag_stdp = 0.00230372048
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
    w = clip(w + 0.2*lr*Apost, 0, wmax)
'''
post_eqs_exci='''
    Apost += dApost
    w = clip(w*(1-0.0005*lr) + lr*Apre, 0, wmax)
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
    w = clip(w + 0.2*lr*Apost, 0, wmax)
'''
post_eqs_inhi='''
    Apost += dApost
    w = clip(w*(1-0.0005*lr) + lr*Apre, 0, wmax)
'''

synapse_model = "w : 1"



n_input, n_output = 28*28,1225

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
        model['input'] = NeuronGroup(N=n_input, model=neuron_eqs_v3_fit_function, threshold='v >= threshold', reset=reset_eqs, refractory='refrac', method='heun',name='input')
        model['input'].noise_activity = False
        model['input'].idx_spk = 1

        # Output layer
        model['output'] = NeuronGroup(N=n_output, model=neuron_eqs_v3_fit_function, threshold='v >= threshold', reset=reset_eqs, refractory='refrac', method='heun',name='output')
        model['output'].noise_activity = False
        model['output'].idx_spk = 1
        
        # Full connection from input to about with excitatory synapses conducted by STDP
        model['input_synapse_exci'] = Synapses(model['input'], model['output'], model=stdp_eqs_exci, on_pre=pre_eqs_exci, on_post=post_eqs_exci,name='input_synapse_exci')
        model['input_synapse_exci'].connect(True)
        model['input_synapse_exci'].lr = 1
        model['input_synapse_exci'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_exci'].delay = 'rand()*0*us'

        # Full connection from input to about with inhibitory synapses conducted by STDP
        model['input_synapse_inhi'] = Synapses(model['input'], model['output'], model=stdp_eqs_inhi, on_pre=pre_eqs_inhi, on_post=post_eqs_inhi,name='input_synapse_inhi')
        model['input_synapse_inhi'].connect(True)
        model['input_synapse_inhi'].lr = 1
        model['input_synapse_inhi'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_inhi'].delay = 'rand()*0*us'

        # Lateral inhibition with synapses set up with random but constant weights.
        model['wta_synapse'] = Synapses(model['output'], model['output'], model=synapse_model, on_pre='gi_post += w',name='wta_synapse')
        model['wta_synapse'].connect(condition='i != j')
        model['wta_synapse'].w = 'rand() * wmax * 3.7 + 0.8*wmax'
        model['wta_synapse'].delay = 'rand()*0*us'

        model['output_SP'] = SpikeMonitor(model['output'], record= False,name='output_SP')


        if (debug):
            model['input_SP'] = SpikeMonitor(model['input'], record=True,name='input_SP')
            model['output_SP'] = SpikeMonitor(model['output'], record=True,name='output_SP')

        print("You've created :",len(model['input_synapse_exci'])+len(model['input_synapse_inhi'])+len(model['wta_synapse']) , "synapses in your network \n \n")

        self.net = Network(model)


    def __getitem__(self,key):
        return self.net[key]

    def training(self,idx_epoch,X_train,y_train,noise=False,plot=False):
        '''
            Training function with synaptic normalization
        '''

        # (un)Activate noise during training
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Activate STDP learning
        self.net['input_synapse_exci'].lr = 1
        self.net['input_synapse_inhi'].lr = 1

        for j, (sample, label) in enumerate(zip(X_train, y_train)):

            self.net['input'].I = sample.ravel() * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the training simulation
            #self.net.run(time_per_sample, report='text', report_period = 60*second)
            self.net.run(time_per_sample)

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net['input'].noise_activity = False
            self.net['output'].noise_activity = False
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest
            self.net['input'].idx_spk = 1
            self.net['output'].idx_spk = 1
            self.net['input'].noise_activity = noise
            self.net['output'].noise_activity = noise


            # Normalization according to the divisive enforcement rule from Goodhill&Barrow (1994) :
            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j] = self.net['input_synapse_exci'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 85/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_exci'].w = weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j]

            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j] = self.net['input_synapse_inhi'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 40/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_inhi'].w = weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j]


    def labelisation(self,idx_epoch,X_train,y_train,noise=False,plot=False):
        '''
            Function for labeling output neurons based on their response to inputs
        '''

        # (un)Activate noise during training
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Desactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes = np.zeros((10, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        label_percent = 0.85 #15% of the previous trained data will be re-used to label output neurons
        nb_label = int(label_percent*len(y_train))
        X_labeling = X_train[nb_label:]
        y_labeling = y_train[nb_label:]

        for j, (sample, label) in enumerate(zip(X_labeling, y_labeling)):

            self.net['input'].I = sample.ravel() * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            #self.net.run(time_per_sample, report='text', report_period = 60*second)
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes[int(label)] += counter_spike

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net['input'].noise_activity = False
            self.net['output'].noise_activity = False
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest
            self.net['input'].idx_spk = 1
            self.net['output'].idx_spk = 1
            self.net['input'].noise_activity = noise
            self.net['output'].noise_activity = noise

            old_spike_counts = np.copy(self.net['output_SP'].count)

        labeled_neurons = np.argmax(spikes, axis=0)
        redundancy_neuron = np.bincount(labeled_neurons, minlength=10)
        self.labeled_neurons = labeled_neurons
        self.redundancy_neuron = redundancy_neuron


    def intermediate_evaluation(self,idx_epoch,X_test,y_test,noise=False,plot=False):
        '''
            Function for testing the network on the test dataset and calculates the training accuracy
        '''
        # (un)Activate the noise
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Unactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes = np.zeros((10, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        num_correct_output_max = 0

        for j, (sample, label) in enumerate(zip(X_test, y_test)):
            self.net['input'].I = sample.ravel() * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes[int(label)] += counter_spike

            # Prediction of the SNN post labelisation
            output_label_max = self.labeled_neurons[np.argmax(counter_spike)]

            # Check prediction of the class of the sample
            if output_label_max == int(label):
                num_correct_output_max += 1

            ## Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net['input'].noise_activity = False
            self.net['output'].noise_activity = False
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest
            self.net['input'].idx_spk = 1
            self.net['output'].idx_spk = 1
            self.net['input'].noise_activity = noise
            self.net['output'].noise_activity = noise

            old_spike_counts = np.copy(self.net['output_SP'].count)


        accuracy_max = (num_correct_output_max / len(X_test))

        # Export synaptic weights
        weight_wta = np.zeros((n_output*(n_output-1)))
        weight_wta = np.copy(self.net['wta_synapse'].w)

        weight_matrix = np.zeros([n_input, n_output])
        weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j] = self.net['input_synapse_exci'].w
        weight_exci = np.copy(weight_matrix)

        weight_matrix = np.zeros([n_input, n_output])
        weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j] = self.net['input_synapse_inhi'].w
        weight_inhi = np.copy(weight_matrix)

        return accuracy_max,self.labeled_neurons,spikes,weight_exci,weight_inhi,weight_wta

    def final_evaluation(self,idx_epoch,X_test,y_test,noise=False,plot=False):
        '''
            Function for testing the post-trained network on the test dataset and calculates the final accuracy
        '''
        # (un)Activate the noise
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Desactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes = np.zeros((10, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        num_correct_output_max = 0
        confusion_matrix = np.zeros(100).reshape(10,10)

        for j, (sample, label) in enumerate(zip(X_test, y_test)):
            self.net['input'].I = sample.ravel() * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes[int(label)] += counter_spike

            # Prediction of the SNN post labelisation
            output_label_max = self.labeled_neurons[np.argmax(counter_spike)]

            # Check prediction of the class of the sample
            if output_label_max == int(label):
                num_correct_output_max += 1
            confusion_matrix[int(label)][output_label_max] +=1

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net['input'].noise_activity = False
            self.net['output'].noise_activity = False
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest
            self.net['input'].idx_spk = 1
            self.net['output'].idx_spk = 1
            self.net['input'].noise_activity = noise
            self.net['output'].noise_activity = noise

            old_spike_counts = np.copy(self.net['output_SP'].count)

        accuracy_max = (num_correct_output_max / len(X_test))

        return accuracy_max,confusion_matrix



directory = 'directory_name'
itera = 0

def main_mnist_analog_sd_noise(iterat,dir_name,noise=False):
    '''
    Runs a simulation of an SNN on the MNIST dataset using eNeuron from scenario 2

    Parameters:
        iterat (int): An identifier to differentiate simulations when the function is called using the multiprocessing library.
        dir_name (str): Name of the directory where the data will be saved.
        noise (bool): Flag to activate inherent noise in the neuron model.
    '''
    
    global directory
    global itera
    directory = dir_name
    itera = iterat
    os.makedirs(f'data_backup_test/{directory}',exist_ok=True)
    os.makedirs(f'data_backup_test/{directory}/{directory+str(itera)}',exist_ok=True)

    accuracy_evol = []

    # Importing mnist data
    X_train_batches, y_train_batches, X_test_batches, y_test_batches, nb_batch = load_mnist(nb_train=60000, nb_test=10000, working_range=3, batch_size_train=1500,batch_size_test=200)

    # Creating the SNN model
    debug = False
    noise = True # activate inherent noise (scenario 2)
    model = SNN_eNeuron(debug=debug)

    epoch_max = 7
    epoch_max = min(epoch_max, nb_batch)

    tic = time.time()

    # Before initiating any training, the network should be tested to establish a baseline performance. 
    # The initial predictions are expected to be random, resulting in an accuracy of approximately 10%
    model.labelisation(-1,X_train_batches[-1], y_train_batches[-1],noise=noise,plot=debug)
    accuracy_max,labeled_neurons,spikes,weight_exci,weight_inhi,weight_wta = model.intermediate_evaluation(-1,X_test_batches[-1], y_test_batches[-1],noise=True,plot=debug)
    accuracy_evol.append(accuracy_max)
    print(" FINISH analog pre-epoch with accuracy = ", accuracy_max," in ",time.time()-tic, "s \n")

    
    #Training 
    tic = time.time()
    for idx_epoch in range(epoch_max):

        model.training(idx_epoch,X_train_batches[idx_epoch], y_train_batches[idx_epoch],noise=noise,plot=debug)
        model.labelisation(idx_epoch,X_train_batches[idx_epoch], y_train_batches[idx_epoch],noise=noise,plot=debug)

        accuracy_max,labeled_neurons,spikes,weight_exci,weight_inhi,weight_wta = model.intermediate_evaluation(idx_epoch,X_test_batches[idx_epoch], y_test_batches[idx_epoch],noise=True,plot=debug)
        accuracy_evol.append(accuracy_max)
        save_accuracy_txtfile_short(f"data_backup_test/{directory}/Accuracy_evolution_save.txt",np.array(accuracy_evol))
        print(" FINISH analog epoch ", idx_epoch," with accuracy = ",accuracy_max," in ",time.time()-tic, "s \n")
        tic = time.time()

        ### Save the synaptic weights and the labeling to be able to rebuild the network to analyze it at each epoch
        with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_exci{idx_epoch}', 'wb') as fp:
            pickle.dump(weight_exci, fp)
        with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_inhi{idx_epoch}', 'wb') as fp:
            pickle.dump(weight_inhi, fp)
        with open(f'data_backup_test/{directory}/{directory+str(itera)}/weightfile_wta{idx_epoch}', 'wb') as fp:
            pickle.dump(weight_wta, fp)
        with open(f'data_backup_test/{directory}/{directory+str(itera)}/labelfile{idx_epoch}', 'wb') as fp:
            pickle.dump(labeled_neurons, fp)


        if accuracy_max >= 0.81:
            # Stopping criterion is implemented because the learning rate for unsupervised STDP is not necessarily positive.
            break 


    csv_save(f'data_backup_test/{directory}/saving_accuracy_evolution.csv',accuracy_evol)
    save_accuracy_txtfile(f"data_backup_test/{directory}/Accuracy_nb_epoch_save.txt", itera, idx_epoch+1, accuracy_max,np.array(accuracy_evol))
    # Plot accuracy evolution
    plot_accuracy_evol(directory,accuracy_evol)

    # Testing the trained network
    X_train_batches, y_train_batches, X_test_batches, y_test_batches, nb_batch = load_mnist(nb_train=5000, nb_test=5000, working_range=3, batch_size_train=5000,batch_size_test=5000)
    accuracy_max,confusion_matrix = model.final_evaluation(idx_epoch,X_test_batches[0], y_test_batches[0],noise=True,plot=debug)
    df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
    plt.figure(figsize=(10,7))
    plt.title("Confusion matrix")
    sn.set(font_scale=1.1) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 11}) 
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f"data_backup_test/{directory}/{directory+str(itera)}/confusion_matrix.png")
    csv_save(f'data_backup_test/{directory}/saving_accuracy_eval_noise_analog.csv',[accuracy_max])

    return accuracy_max
