
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

Here, the analog SNN is tested on the XOR benchmark. 
The analog SNN has been trained with noiseless neurons and it is tested with neuron models presenting 
intrinsic random noise (SCENARIO 3.2 in the paper)

This code employs the architecture : [2,13]+exci+inhi+wta.
It means that there is 2 input neurons fully connected with 13 outputs neurons
with both excitatory and inhibitory synapses conducted with STDP rules. Plus, there is a lateral inhibition
playing a soft-Winner-Take-All synaptic connection between output neurons. It implies 208 synapses. 

It operates and returns:
    - Creates XOR dataset in batches.
    - Initializes and configures the analog SNN model from a trained network
    - Performs a final evaluation of the trained networt.

Example usage:
    This function is called in 'XOR_without_noise.py'

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

threshold = v_threshold + int(noise_activity)*(int(I_syn>frequency_min)*sqrt(scale*q/(I_total+avoid_error)*(1/(I_syn+frequency_min) + R_ds*C_m/pi)))*randn()*(v_threshold-v_reset)*I_syn*sqrt(idx_spk) : volt (constant over dt)
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
    def __init__(self, labeled_neurons,weight_exci,weight_inhi,weight_wta,debug=False):
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
        model['input'].idx_spk = 1
        model['input'].noise_activity = True

        # Output layer
        model['output'] = NeuronGroup(N=n_output, model=neuron_eqs_v3_interp_function, threshold='v >= threshold', reset=reset_eqs, refractory='refrac', method='heun',name='output')
        model['output'].idx_spk = 1
        model['output'].noise_activity = True

        # Full connection from input to about with excitatory synapses conducted by STDP
        model['input_synapse_exci'] = Synapses(model['input'], model['output'], model=stdp_eqs_exci, on_pre=pre_eqs_exci, on_post=post_eqs_exci,name='input_synapse_exci')
        model['input_synapse_exci'].connect(True)
        model['input_synapse_exci'].lr = 1
        model['input_synapse_exci'].w = weight_exci
        model['input_synapse_exci'].delay = 'rand()*0*us'

        # Full connection from input to about with inhibitory synapses conducted by STDP
        model['input_synapse_inhi'] = Synapses(model['input'], model['output'], model=stdp_eqs_inhi, on_pre=pre_eqs_inhi, on_post=post_eqs_inhi,name='input_synapse_inhi')
        model['input_synapse_inhi'].connect(True)
        model['input_synapse_inhi'].lr = 1
        model['input_synapse_inhi'].w = weight_inhi
        model['input_synapse_inhi'].delay = 'rand()*0*us'

        # Lateral inhibition with synapses set up with random but constant weights.
        model['wta_synapse'] = Synapses(model['output'], model['output'], model=synapse_model, on_pre='gi_post += w',name='wta_synapse')
        model['wta_synapse'].connect(condition='i != j')
        model['wta_synapse'].w = weight_wta
        model['wta_synapse'].delay = 'rand()*0*us'

        model['output_SP'] = SpikeMonitor(model['output'], record=False,name='output_SP')

        if (debug):
            model['input_SP'] = SpikeMonitor(model['input'], record=True,name='input_SP')
            model['output_SP'] = SpikeMonitor(model['output'], record=True,name='output_SP')

        print("You've created :",len(model['input_synapse_exci'])+len(model['input_synapse_inhi'])+len(model['wta_synapse']) , "synapses in your network")

        self.net = Network(model)
        self.labeled_neurons = labeled_neurons
        

    def __getitem__(self,key):
        return self.net[key]

    def evaluation(self,X_test,y_test,noise=False,plot=False):
        '''
            Function for testing the post-trained network on the test dataset and calculates the final accuracy
        '''
        # (un)Activate the noise
        self.net['input'].noise_activity = noise
        self.net['output'].noise_activity = noise

        # Unactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes = np.zeros((2, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        num_correct_output_max = 0
      
        for j, (sample, label) in enumerate(zip(X_test, y_test)):
            self.net['input'].I = from_bit_to_current(sample) * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes[from_bit_to_label_2(sample)] += counter_spike    
            
            # Prediction of the SNN post labelisation
            output_label_max = self.labeled_neurons[np.argmax(counter_spike)]

            # Check prediction of the class of the sample
            if output_label_max == int(label):
                num_correct_output_max += 1
        

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

        return accuracy_max

    def plot_3D_XOR(self,n_point=5,noise=False,plot=False):
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
            output_label_max2 = self.labeled_neurons[np.argmax(counter_spike)]
            Z_3D.append(output_label_max2)
            
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

        return X_3D,Y_3D,Z_3D



def evaluation_noiseless_and_analog(labeled_neurons,weight_exci,weight_inhi,weight_wta):

    print(" STARTING assessing_without_noise_and_analog: \n")
    tic = time.time()

    # Creating and Testing the SNN model from pretrained network
    debug = False
    noise = True
    model = SNN_eNeuron(labeled_neurons,weight_exci,weight_inhi,weight_wta,debug=debug)

    n_train,n_test = 25,25
    X_train,y_train,X_test,y_test = batch_data(n_train,n_test)
    accuracy_max = model.evaluation(X_test,y_test,noise=noise,plot=debug)

    #3D figure
    X_3D,Y_3D,Z_3D = model.plot_3D_XOR(n_point=10,noise=noise,plot=False)

    print(" DONE with assessing_without_noise_and_analog with noise = ", noise, "in ",time.time()-tic, "s \n")

    return accuracy_max,Z_3D