# --------------------------------------------------------------
# Name: Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits
# Author: Yannaël Bossard
# Date: 2024-03-23
# Source: https://github.com/YannaelB 
# Description: 
'''
The code below aims to find the best number of output neurons. The method employed is very simple. It simulates eSNNs
with number of output neurons as parameters (from 6 to 16, no more because we also want the minimum number 
of neurons and synapses). By analysing the number of output neurons providing the fastest converging on average, It helps to 
decide the final architecture of the eSNN. 3 different criteria of accuracy are presented in the algorithm. Only 
the 'most-spiking-single-neuron' criterion is used.


It operates and returns:
    - Creates necessary directories for data storage.
    - Creates XOR dataset in batches.
    - Initializes and configures the eSNN model for a given number of output neurons
    - Trains the eSNN model over getting 100% accuracy or 100 epochs. 
    - Repeat the processus and saves the number of epoch .

Example usage:
    average_epoch, n_output = assessing_network(13)

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
import time
import warnings
from tqdm import tqdm
import random
from random import shuffle





def batch_data(n_train,n_test):
    X_train = n_train*[[0,0]] + n_train*[[0,1]] + n_train*[[1,0]] + n_train*[[1,1]]
    shuffle(X_train)
    y_train = [a ^ b for a, b in X_train]

    X_test = n_test*[[0,0]] + n_test*[[0,1]] + n_test*[[1,0]] + n_test*[[1,1]]
    shuffle(X_test)
    y_test = [a ^ b for a, b in X_test]
    
    return X_train,y_train,X_test,y_test

def from_bit_to_current(x):
    if x == [0,0]:
        return [0.3,0.3]
    elif x == [0,1]:
        return [0.3,2]
    elif x == [1,0]:
        return [2,0.3]
    elif x == [1,1]:
        return [2,2]
    else:
        print(" WARNING PROBLEM !")
        
def from_bit_to_label_2(x):
    if x == [0,0]:
        return 0
    elif x == [0,1]:
        return 1
    elif x == [1,0]:
        return 1
    elif x == [1,1]:
        return 0
    else:
        print(" WARNING PROBLEM !")
        
def from_bit_to_label_4(x):
    if x == [0,0]:
        return 0
    elif x == [0,1]:
        return 1
    elif x == [1,0]:
        return 2
    elif x == [1,1]:
        return 3
    else:
        print(" WARNING PROBLEM !")
        
def from_label_to_bit_4(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x == 2:
        return 1
    elif x == 3:
        return 0
    else:
        print(" WARNING PROBLEM !")




def weight_evolution(weight_evol_exci,weight_evol_inhi,n_output):
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1, 2, 1)  
    for idx_w in range(len(weight_evol_exci[0])):
        weight = [ligne[idx_w] for ligne in weight_evol_exci]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')  
    plt.title('Evolution des poids synaptiques EXCI ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.ylim(-wmax-1, wmax+1) 
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for idx_w in range(len(weight_evol_inhi[0])):
        weight = [ligne[idx_w] for ligne in weight_evol_inhi]
        plt.plot(range(len(weight)), weight, label=f'w{idx_w}')  
    plt.title('Evolution des poids synaptiques INHI ')
    plt.xlabel(f' epoch ')
    plt.ylabel(' weight [0,1]')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"results/weight_evolution_epoch_{n_output}output.png")


def plot_accuracy_evol(accuracy_evol,n_output):
    plt.figure(figsize=(10, 6)) 
    plt.plot(range(len(accuracy_evol[0])),accuracy_evol[0],label="max criteria")
    plt.plot(range(len(accuracy_evol[1])),accuracy_evol[1],label="max mean criteria")
    plt.plot(range(len(accuracy_evol[2])),accuracy_evol[2],label="first spike criteria")
    plt.title(' accuracy evolution ')
    plt.xlabel(f' epoch  ')
    plt.ylabel(' accuracy ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/accuracy_evol_epoch_{n_output}output.png")





from brian2 import clear_cache
from brian2 import device
device.reinit()
start_scope()

# Simulation parameters
defaultclock.dt = 0.005*us  
time_per_sample = 100 * us
resting_time = 20 * us

#Model parameters
v_rest = -80*mV
v_threshold = 60*mV  
v_reset = -100*mV
v0 = -40*mV  
tau = 0.5*ms
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

n_input, n_output = 2,14

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


class SNN_eNeuron():
    def __init__(self, debug=False, pixel_intensity=8):

        from brian2 import clear_cache
        from brian2 import device
        device.reinit()
        start_scope()

        # Simulation parameters
        defaultclock.dt = 0.005*us  

        model = {}

        model['input'] = NeuronGroup(N=n_input, model=neuron_eqs_v1, threshold='v >= v_threshold', reset='v = v_reset', refractory='refrac', method='heun',name='input')
        model['input'].inhi_factor = 1
        model['input'].activity = 1

        print("n_output = ",n_output)
        model['output'] = NeuronGroup(N=n_output, model=neuron_eqs_v1, threshold='v >= v_threshold', reset='v = v_reset', refractory='refrac', method='heun',name='output')
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

            # Observation weight evolution
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

        label_percent = 0.5 #50% of the previous trained data will be re-used to label output neurons
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

            # Observation weight evolution
            weight = [self.net['input_synapse_exci'].w[i] for i in range(len(self.net['input_synapse_exci'].w))]
            self.weight_evol_exci.append(weight)
            weight = [self.net['input_synapse_inhi'].w[i] for i in range(len(self.net['input_synapse_inhi'].w))]
            self.weight_evol_inhi.append(weight)


        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)  
            label_list = [[0],[1]]
            for i in range(len(spikes2)):
                if spikes2[i].any() != np.zeros(len(spikes2[i])).any():
                    plt.plot(range(len(spikes2[i])),spikes2[i], label=f'label {label_list[i]}')
            plt.legend()

            plt.subplot(1, 2, 2)  
            label_list = [[0,0],[0,1],[1,0],[1,1]]
            for i in range(len(spikes4)):
                if spikes4[i].any() != np.zeros(len(spikes4[i])).any():
                    plt.plot(range(len(spikes4[i])),spikes4[i], label=f'label {label_list[i]}')
            plt.legend()
            plt.show()
        
        
        labeled_neurons4 = np.argmax(spikes4, axis=0)
        labeled_neurons2 = np.argmax(spikes2, axis=0)
        redundancy_neuron2 = np.bincount(labeled_neurons2, minlength=2)
        redundancy_neuron4 = np.bincount(labeled_neurons4, minlength=5)
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1) 
            bar(range(len(redundancy_neuron4)),redundancy_neuron4)
            xticks(range(len(redundancy_neuron4)))

            plt.subplot(1, 2, 2)  
            bar(range(len(redundancy_neuron2)),redundancy_neuron2)
            xticks(range(len(redundancy_neuron2)))
            plt.show()
        
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
            
            output_label_max4 = from_label_to_bit_4(self.labeled_neurons4[np.argmax(counter_spike)])
            output_label_mean4 = from_label_to_bit_4(np.argmax(classifi_matrix4))
            output_label_temporal4 = from_label_to_bit_4(self.labeled_neurons4[self.net['output_SP'].i[0]])
            
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

def save_accuracy_txtfile(filename, n_output, mean_epoch,mean_epoch_vect):
    with open(filename, 'a') as file:
        file.write(f"\n Method unsupervised with  {n_output} output neurons: \n")
        file.write(" It has taken on average = "+ str(mean_epoch) +" epoch for reaching 100'%' accuracy \n")
        mean_epoch_vect_str = np.array_str(mean_epoch_vect)
        file.write(mean_epoch_vect_str)
        file.write("\n")
        file.write("\n")

def assessing_network(n_output_):

    global n_output 
    n_output = n_output_

    epoch = 100
    amount_assessing = 100

    accuracy_evol = [[],[],[]]
    n_train,n_test = 10,10 #It means one epoch is composed of 10*[[0,0],[0,1],[1,0],[1,1]] (shuffled)
   
    nb_epoch_accuracy100 = []
    for i_assess in tqdm(range(amount_assessing)):
        seed()
        # Creating the SNN model
        debug = False
        model = SNN_eNeuron(debug=debug) 

        for idx_epoch in range(epoch):
            # Creating batch data
            X_train,y_train,X_test,y_test = batch_data(n_train,n_test)

            model.training(idx_epoch,X_train,y_train,plot=debug)
            model.labelisation(idx_epoch,X_train,y_train,plot=debug)

            accuracy_max,accuracy_mean,accuracy_temporal,accuracy_2,accuracy_4,weight_evol_exci,weight_evol_inhi = model.evaluation(idx_epoch,X_test,y_test,plot=debug)
            accuracy_evol[0].append(accuracy_max) #most-spiking-single-neuron criterion
            accuracy_evol[1].append(accuracy_mean) #most-spiking-group-neurons criterion
            accuracy_evol[2].append(accuracy_temporal) #first-spiking-single-neuron criterion

            if accuracy_max >= 1:
                break
        print("done in idx_epoch = ", idx_epoch+1, "epochs for accuracy = ", accuracy_max)
        nb_epoch_accuracy100.append(idx_epoch+1)

        if i_assess%18 == 0:
            # Plot weight evolution
            weight_evolution(weight_evol_exci,weight_evol_inhi,n_output)

        save_accuracy_txtfile(f"results/Accuracy_nb_epoch_save{n_output}.txt", n_output, np.mean(nb_epoch_accuracy100),np.array(nb_epoch_accuracy100))

    # Plot accuracy evolution
    plot_accuracy_evol(accuracy_evol,n_output)

    return np.mean(nb_epoch_accuracy100),n_output_
    



