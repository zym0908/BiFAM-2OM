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
## Running
## Project Structure
- `data/`: Contains the required datasets, specifically iRNA-ac4C and Meta-ac4C.
- `Result/`: Stores pre-trained models ready for inference. It includes TBC-ac4C.pt.

In addition, the main scripts and files are as follows:
- `MyDataset.py`: Handles data loading and preprocessing.
- `until.py`: Computes and evaluates model performance metrics.
- `Mymodel.py`: Defines the BiFAM-2OM model architecture.
- `Mytrain.py`: Training script can be run directly to train the model.
- `Mytest.py`: Testing script can be run directly to evaluate the model and reproduce results.

If you aim to train the BiFAM-2OM model or use a successfully trained model for testing, please run the following code:
```bash
python Mytrain.py
python Mytest.py
```
