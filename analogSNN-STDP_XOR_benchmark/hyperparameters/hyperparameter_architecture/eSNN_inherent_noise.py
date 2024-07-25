
# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-03-23
# Source: https://github.com/YannaelB 
# Description: 
'''
This code is executed in 'main_training.py' and is part of the algorithm for optimisation of the eSNN's architecture.
This is a simple algorithm running different proposed architectures and saving the number of epoch required
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
from tqdm import tqdm
import random
from random import shuffle
from tools import *

from brian2 import clear_cache
from brian2 import device
device.reinit()
start_scope()


# Simulation parameters
defaultclock.dt = 0.005*us  
time_per_sample = 100 * us
resting_time = 20 * us

# Defining the linear interpolation of the eNeuron ML behavior
i_new, f_new = extracting_eNeuron_behavior(csv_file_path='FT_augmented.csv',nb_point_interpolation=20)
interp_function = interp1d(i_new, f_new, fill_value='extrapolate')

@check_units(x=amp, result = Hz)
def y_interp_function(x):
    return interp_function(x)*Hz

#Model parameters
v_rest = -80*mV
v_threshold = 60*mV  
v_std_thresh = 5*mV 
v_reset = -100*mV
v0 = -40*mV  
tau = 500*us
tau2 = 15*us
Rm = 5000*Mohm
I0=1*amp
a_v = 1126.6343973469338
b_v = 169200.45773494235
x_threshold_v= 8.3e-11*amp
avoid_error = 0.01*pA
a_value = 6.23667974e13
refrac = 0*us
alpha_decay = 1
alpha_e = 0.15
alpha_i = 0.04
beta_e = 1.9
beta_i = 0.5
avoid_error = 0.01*pA
c = 1


# This model use a constant Resistance and a log approximation
neuron_eqs_v1 = '''
dv/dt =  (v_rest - v) / tau2 + (( int((I + Ie+I_inhi) > x_threshold_v) *((a_v + b_v * log((1-int((I + Ie+I_inhi) > x_threshold_v))+int((I + Ie+I_inhi) > x_threshold_v)*(I+Ie+I_inhi+avoid_error)/(x_threshold_v) ))*I0)/a_value)*Rm) / tau : volt (unless refractory)
Ie =  activity*alpha_e*ge * uS *(80*mV -c*v)*inhi_factor           : amp
I_inhi =  activity*alpha_i*gi * uS *(-120*mV -c*v)           : amp
I : amp
dge/dt = -ge/(beta_e*us)            : 1
dgi/dt = -gi/(beta_i*us)            : 1
inhi_factor : 1
activity :1
'''



#STDP parameters
tau_stdp = 1.88116837
mag_stdp = 0.007070372048
taupre = tau_stdp*us
taupost = taupre
wmax = 1
dApre = mag_stdp
dApost = -dApre * taupre / taupost * 1.05

stdp_eqs_exci = '''
    w : 1
    alpha : 1 (shared)
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
'''
pre_eqs_exci='''
    ge_post += w
    Apre += dApre
    w = clip(alpha*w + lr*Apost, 0, wmax)
'''
post_eqs_exci='''
    Apost += dApost
    w = clip(w + lr*Apre, 0, wmax)
'''

stdp_eqs_inhi = '''
    w : 1
    alpha : 1 (shared)
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
'''
pre_eqs_inhi='''
    gi_post += w
    Apre += dApre
    w = clip(alpha*w + lr*Apost, 0, wmax)
'''
post_eqs_inhi='''
    Apost += dApost
    w = clip(w + lr*Apre, 0, wmax)
'''

synapse_model = "w : 1"


n_input, n_output = 2,13

class SNN_eNeuron():
    def __init__(self, debug=False):
        '''
            Initialization including setting up neuron groups and synapses
        '''

        from brian2 import clear_cache
        from brian2 import device
        device.reinit()
        start_scope()

        # Simulation parameters
        defaultclock.dt = 0.01*us  

        model = {}

        model['input'] = NeuronGroup(N=n_input, model=neuron_eqs_v1, threshold='v >= v_threshold + v_std_thresh*randn()', reset='v = v_reset', refractory='refrac', method='heun',name='input')
        model['input'].inhi_factor = 1
        model['input'].activity = 1

        model['output'] = NeuronGroup(N=n_output, model=neuron_eqs_v1, threshold='v >= v_threshold + v_std_thresh*randn()', reset='v = v_reset', refractory='refrac', method='heun',name='output')
        model['output'].inhi_factor = 1
        model['output'].activity = 1

        model['input_synapse_exci'] = Synapses(model['input'], model['output'], model=stdp_eqs_exci, on_pre=pre_eqs_exci, on_post=post_eqs_exci,name='input_synapse_exci')
        model['input_synapse_exci'].connect(True)
        model['input_synapse_exci'].lr = 0.5
        model['input_synapse_exci'].alpha = alpha_decay
        model['input_synapse_exci'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_exci'].delay = 'rand()*0*us'

        model['input_synapse_inhi'] = Synapses(model['input'], model['output'], model=stdp_eqs_inhi, on_pre=pre_eqs_inhi, on_post=post_eqs_inhi,name='input_synapse_inhi')
        model['input_synapse_inhi'].connect(True)
        model['input_synapse_inhi'].lr = 0.5
        model['input_synapse_inhi'].alpha = alpha_decay
        model['input_synapse_inhi'].w = 'rand() * wmax * 0.8 + 0.1*wmax'
        model['input_synapse_inhi'].delay = 'rand()*0*us'

        model['wta_synapse'] = Synapses(model['output'], model['output'], model=synapse_model, on_pre='gi_post += w',name='wta_synapse')
        model['wta_synapse'].connect(condition='i != j')
        model['wta_synapse'].w = 'rand() * wmax * 1.2 + 0.8*wmax'
        model['wta_synapse'].delay = 'rand()*0*us'

        model['output_SP'] = SpikeMonitor(model['output'], record=True,name='output_SP')

        if (debug):
            model['input_SP'] = SpikeMonitor(model['input'], record=True,name='input_SP')
            model['output_SP'] = SpikeMonitor(model['output'], record=True,name='output_SP')

        print("You've created :",len(model['input_synapse_exci'])+len(model['input_synapse_inhi'])+len(model['wta_synapse']) , "synapses in your network")

        self.net = Network(model)
        
        self.weight_evol_exci = []
        self.weight_evol_inhi = []

    def __getitem__(self,key):
        return self.net[key]

    def training(self,idx_epoch,X_train,y_train,plot=False):
        '''
            Training function with synaptic normalization and weight observation
        '''

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


            #NORMALISATION :
            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j] = self.net['input_synapse_exci'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 1/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_exci'].w = weight_matrix[self.net['input_synapse_exci'].i, self.net['input_synapse_exci'].j]

            weight_matrix = np.zeros([n_input, n_output])
            weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j] = self.net['input_synapse_inhi'].w
            sumCol = np.sum(weight_matrix, axis=0)
            colFactors = 0.5/sumCol
            for jj in range(n_output):
                weight_matrix[:,jj] *= colFactors[jj]
            self.net['input_synapse_inhi'].w = weight_matrix[self.net['input_synapse_inhi'].i, self.net['input_synapse_inhi'].j]

        #Observation weight evolution
        weight = [self.net['input_synapse_exci'].w[i] for i in range(len(self.net['input_synapse_exci'].w))]
        self.weight_evol_exci.append(weight)
        weight = [self.net['input_synapse_inhi'].w[i] for i in range(len(self.net['input_synapse_inhi'].w))]
        self.weight_evol_inhi.append(weight)


    def labelisation(self,idx_epoch,X_train,y_train,plot=False):
        '''
            Function for labeling output neurons based on their response to inputs
        '''
        
        # Desactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes2 = np.zeros((2, n_output))
        spikes4 = np.zeros((4, n_output))

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
            spikes4[from_bit_to_label_4(sample)] += counter_spike 

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            old_spike_counts = np.copy(self.net['output_SP'].count)

        #Observation weight evolution
        weight = [self.net['input_synapse_exci'].w[i] for i in range(len(self.net['input_synapse_exci'].w))]
        self.weight_evol_exci.append(weight)
        weight = [self.net['input_synapse_inhi'].w[i] for i in range(len(self.net['input_synapse_inhi'].w))]
        self.weight_evol_inhi.append(weight)

        
        labeled_neurons4 = np.argmax(spikes4, axis=0)
        labeled_neurons2 = np.argmax(spikes2, axis=0)
        redundancy_neuron2 = np.bincount(labeled_neurons2, minlength=2)
        redundancy_neuron4 = np.bincount(labeled_neurons4, minlength=5)
        self.labeled_neurons2 = labeled_neurons2
        self.redundancy_neuron2 = redundancy_neuron2
        self.labeled_neurons4 = labeled_neurons4
        self.redundancy_neuron4 = redundancy_neuron4

        
    def evaluation(self,idx_epoch,X_test,y_test,plot=False):
        '''
            Evaluation function that tests the trained network on test data and calculate accuracy
        '''

        # Desactivate STDP plasticity
        self.net['input_synapse_exci'].lr = 0
        self.net['input_synapse_inhi'].lr = 0

        spikes2 = np.zeros((2, n_output))
        spikes4 = np.zeros((4, n_output))

        old_spike_counts = np.zeros(n_output)
        old_spike_counts = np.copy(self.net['output_SP'].count)

        num_correct_output_max2 = 0
        num_correct_output_mean2 = 0
        num_correct_output_temporal2 = 0
        
        num_correct_output_max4 = 0
        num_correct_output_mean4 = 0
        num_correct_output_temporal4 = 0

        classifi_matrix2 = np.zeros((2,1))
        classifi_matrix4 = np.zeros((2,1))

        accuracy_2 = np.zeros((2,1))
        accuracy_4 = np.zeros((4,1))

        for j, (sample, label) in enumerate(zip(X_test, y_test)):
            self.net['input'].I = from_bit_to_current(sample) * nA
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            # Start the simulation
            self.net.run(time_per_sample)
            counter_spike = self.net['output_SP'].count - old_spike_counts
            spikes2[from_bit_to_label_2(sample)] += counter_spike    
            spikes4[from_bit_to_label_4(sample)] += counter_spike 
            

            for k in range(len(counter_spike)):
                classifi_matrix2[self.labeled_neurons2[k],0] += counter_spike[k]/self.redundancy_neuron2[self.labeled_neurons2[k]] #normalisation to equilibrate each labelled population


            # Prediction of the SNN post labelisation
            output_label_max2 = self.labeled_neurons2[np.argmax(counter_spike)]
            output_label_mean2 = np.argmax(classifi_matrix2)
            output_label_temporal2 = self.labeled_neurons2[self.net['output_SP'].i[0]]
            
            output_label_max4 = from_label_to_XOR(self.labeled_neurons4[np.argmax(counter_spike)])
            output_label_mean4 = from_label_to_XOR(np.argmax(classifi_matrix4))
            output_label_temporal4 = from_label_to_XOR(self.labeled_neurons4[self.net['output_SP'].i[0]])

            if output_label_max2 == int(label):
                num_correct_output_max2 += 1
                accuracy_2[from_bit_to_label_2(sample)] += 2/len(X_test)
            if output_label_mean2 == int(label):
                num_correct_output_mean2 += 1
            if output_label_temporal2 == int(label):
                num_correct_output_temporal2 += 1
                
            if output_label_max4 == int(label):
                num_correct_output_max4 += 1
                accuracy_4[from_bit_to_label_4(sample)] += 4/len(X_test)
            if output_label_mean4 == int(label):
                num_correct_output_mean4 += 1
            if output_label_temporal4 == int(label):
                num_correct_output_temporal4 += 1

            # Allow the variables to return to their resting values
            self.net['input'].I = 0 * nA
            self.net.run(resting_time) 
            self.net['input'].v = v_rest
            self.net['output'].v = v_rest

            old_spike_counts = np.copy(self.net['output_SP'].count)


        accuracy_max = (num_correct_output_max2 / len(X_test))
        accuracy_mean = (num_correct_output_mean2 / len(X_test))
        accuracy_temporal = (num_correct_output_temporal2 / len(X_test))


        return accuracy_max,accuracy_mean,accuracy_temporal,accuracy_2,accuracy_4, self.weight_evol_exci, self.weight_evol_inhi



directory = "hyperparams_inherent_noise_backup"
def assessing_inherent_noise(itera):
    '''
        Function that assesses the network by training, labeling, evaluating, and plotting results for multiple iterations
    '''

    epoch_max = 100
    os.makedirs(f'results/{directory}',exist_ok=True)

    accuracy_evol = [[],[],[]]
    n_train,n_test = 10,2 #It means one epoch is composed of 10*[[0,0],[0,1],[1,0],[1,1]] (shuffled)
    
    seed()
    # Creating the SNN model
    debug = False
    model = SNN_eNeuron(debug=debug) 

    #Training and Assesing
    for idx_epoch in range(epoch_max):
        # Creating batch data
        X_train,y_train,X_test,y_test = batch_data(n_train,n_test)

        model.training(idx_epoch,X_train,y_train,plot=debug)
        model.labelisation(idx_epoch,X_train,y_train,plot=debug)

        accuracy_max,accuracy_mean,accuracy_temporal,accuracy_2,accuracy_4,weight_evol_exci,weight_evol_inhi = model.evaluation(idx_epoch,X_test,y_test,plot=debug)
        accuracy_evol[0].append(accuracy_max)
        accuracy_evol[1].append(accuracy_mean)
        accuracy_evol[2].append(accuracy_temporal)

        if accuracy_max >= 1:
            break
    print("done in idx_epoch = ", idx_epoch+1, "epochs for accuracy = ", accuracy_max)
    csv_save(f'results/{directory}/saving_nb_epoch_converging.csv',[idx_epoch+1])
    csv_save(f'results/{directory}/saving_accuracy_max.csv',accuracy_evol[0])
    csv_save(f'results/{directory}/saving_accuracy_mean.csv',accuracy_evol[1])
    csv_save(f'results/{directory}/saving_accuracy_temporal.csv',accuracy_evol[2])


    if itera%17 == 0:
        # Plot weight evolution
        weight_evolution(directory,weight_evol_exci,weight_evol_inhi,n_output)

    save_accuracy_txtfile(f"results/{directory}/Accuracy_nb_epoch_save.txt", n_output, idx_epoch+1, accuracy_max,np.array([accuracy_max,accuracy_mean,accuracy_temporal]))

    return idx_epoch+1



#assessing_inherent_noise(1)