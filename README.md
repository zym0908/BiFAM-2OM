# BiFAM-2OM
Bidirectional Fusion of Adaptive Attention-Enhanced Multi-Scale Convolution and BiLSTM for RNA 2’-O-Methylation Sites Prediction
## Description
BiFAM-2OM is a novel bioinformatics tool designed to accurately predict 2'-O-methylation sites in RNA. The method integrates a Transformer encoder, a multi-scale convolutional attention mechanism, and a bidirectional long short-term memory (BiLSTM) network, and employs a bidirectional cross-attention mechanism to weight and fuse features. This model is capable of accurately identifying 2'-O-methylation sites across all four nucleotide types.
## Requirements
Create and activate the Conda virtual environment before running:
```bash
conda create -n bifam-2om python=3.7
conda activate bifam-2om
```
Once the environment is set up, install the packages required for the project:
```bash
pip install torch==1.13.1 torchvision==0.14.1 numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2 termcolor==1.1.0
```
## Project Structure
The folder and file composition of the project is shown below:
- `data/`: The dataset is divided into four subsets—A, U, C, and G—based on the base d at the modified site, and each subset is split into training and testing sets in a 7:3 ratio.
- `Result/`: Store the trained model files corresponding to the four datasets

- `MyDataset.py`: Implement RNA sequence encoding, dataset loading and splitting, and output training and test sets using DataLoader.
- `until.py`: Provides model evaluation, metric calculation, and loss functions.
- `Mymodel.py`: Model structure of RNA 2'-O-methylation sites
- `Mytrain.py`: Run this file to train the model.
- `Mytest.py`: Run this file to test the trained model.
## Quick Start
Execute the following commands to complete model training and evaluation:
```bash
python Mytrain.py
python Mytest.py
```
