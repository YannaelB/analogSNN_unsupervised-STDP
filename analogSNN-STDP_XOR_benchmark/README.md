
## Analog Spiking Neuron Model for Unsupervised STDP-based learning in Neuromorphic Circuits

This directory contains code and resources for the Analog Spiking Neuron Model for Unsupervised STDP-based learning in neuromorphic circuits, specifically applied to the XOR problem.

### Directory Structure

- **data_backup**: Contains CSV-backup files from simulations and pickle files of the trained networks.
- **hyperparameters**: Contains scripts unabling to optimize the hyperparameters of the analog SNN's architecture
- **notebooks**: Contains Jupyter notebooks that serve as visual tutorials for the functions present in the repository.
- **results**: Contains results from processed data.

### Files

- `main_training_XOR.py`: Main script for training the analog SNN model.
- `main_processing_accuracy_XOR.py`: Main script for processing the data from training and testing of the analog SNN.
- `main_processing_3Dvisu_XOR.py`: Main script for processing the data from testing of the analog SNN.
- `XOR_without_noise.py`: Script for training/testing the analog SNN (intrinsic random noise related to scenario 3) using the XOR problem as a benchmark.
- `XOR_cst_sd_noise.py`: Script for training/testing the analog SNN (intrinsic random noise related to scenario 1) using the XOR problem as benchmark.
- `XOR_analog_sd_noise.py`: Script for training/testing the analog SNN (intrinsic random noise related to scenario 2) using the XOR problem as benchmark.
- `XOR_analog_eval.py`: Script for testing the trained analog SNN (intrinsic random noise related to scenario 3.2) using the XOR problem as benchmark.
- `tools_XOR.py`: Script providing a collection of utility functions to support the previous scripts.
-`FT_augmented.csv`: Activation function of the ML eNeuron from PSL results



### Running the Code

First, clone the repository and install the required python packages from `requirements.txt`.

To train the analog SNN model with different scenarios, use the previous python scripts. Example usage are provided.


