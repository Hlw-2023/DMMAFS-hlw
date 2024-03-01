# DMMAFS
protein function prediction based on multi-modal multi-attention fusion features
This repository contains script which were used to build and train the DMMAFS model together with the scripts for evaluating the model's performance.

# Dependency
torch 1.10.2 python3.6 cuda

# Content
fine_yune.py:Please use this file to train the sample data
protest.py:Please use this file to test the data
network.py:This file contains the model architecture of the network

# Usage
Input command python3.6 fine_tube.py to run the DMMAFS model
Input command python3.6 protest.py to test the DMMAFS model

# step 1:
The fine_tune.py is called to train our model DMMAFS. Please set the "devices" parameter to set which cuda to run on; please set the "type: parameter and "sepci" parameter to set the current training task and the sample species to be trained, respectively.

# step 2:
Please set the path where the model files are saved and update the file path in "model.py", which will hold the iterative model as well as the final best model from model training.

# step 3:
Please update the model path in "protest.py" based on the customized best model data and set the "type" parameter, "speci" parameter, and "device" parameter to set the task category that currently needs to be tested, the species that currently needs to be tested, and the cuda on which "protest.py" currently needs to be run, respectively.

# step 4:
Evaluating DMMAFS via "python3.6 protest.py"

# Related database:
The two databases used to support our work are shown below:
PDB database: http://www.rcsb.org/
Uniprot database: http://www.uniprot.org/


