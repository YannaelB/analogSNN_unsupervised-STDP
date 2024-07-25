
## Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits

This directory contains code and resources for the Analog Spiking Neuron Model for Unsupervised STDP-based learning in neuromorphic circuits, specifically applied to the modeling of the eNeuron.

### Directory Structure

- **results**: Contains results from processed data.

### Files

- `eNeuron_model_paper.ipynb`: Visualizing the model of the ML eNeuron from post-layout simulation resutls.
- `eNeuron_noise_modeling.ipynb`: Visualizing implementation of the intrinsic random noise from the eNeuron.
- `eNeuron_model_paper_fit-function.ipynb`: Visualizing the model of the ML eNeuron from post-layout simulation resutls. It employs a logarithm-fit function rather than interpolation function. It allows to reduce time-computation.
-`FT_augmented.csv`: Activation function of the ML eNeuron from PSL results
-`FT.csv`: Activation function of the ML eNeuron from PSL results
-`spike.csv`: Mmebrane potential, depending on time, of the ML eNeuron from PSL results


### Running the Code

Clone the repository and install the required python packages from `requirements.txt`.



