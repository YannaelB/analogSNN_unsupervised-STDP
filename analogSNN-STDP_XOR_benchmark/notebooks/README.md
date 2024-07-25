
## Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits

This directory contains code and resources for the Analog Spiking Neuron Model for Unsupervised STDP-based learning in neuromorphic circuits, specifically applied to the MNIST dataset.

### Directory Structure

### Files

These notebooks are simplified and visual versions of the codes used for the research article. Please note, these notebooks are drafts and may contain typos. Their purpose is mainly educational. This Jupyter notebook provides an interactive and visual exploration of the initial steps involved in training and testing an eSNN on the XOR problem. It is designed to help users understand and observe the state and variables throughout the training and testing processes. For instance, we can directly observe the impact of the 'defaultclock.dt' value which is crucial for precision and timing-consumption within `XOR_observation_states.ipynb`


- `XOR_rate_code_unsupervised_STDP_lateralWTA.ipynb`:  Visualizing Training and Testing Processes.
- `XOR_rate_code_unsupervised_STDP_hidden_layer_WTA.ipynb`:  Visualizing Training and Testing Processes with a hidden layer
- `XOR_observation_states.ipynb`: Visualizing Training and Testing Processes with a particulaer interest on the variable states of the eNeuron
- `XOR_rate_code_unsupervised_STDP_NOR_NAND_problem.ipynb`:  Visualizing the capacity to solve other logical operator than XOR.
- `XOR_rate_code_unsupervised_STDP_eSynapse.ipynb`:  Visualizing the effect of an electronic synapse model on the eSNN.


