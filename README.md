# Effective Sparsification of Neural Networks with Global Sparsity Constraint

## Requirements:

```
Pytorch 1.4
Python 3.7.7
CUDA Version 10.1
Tensorboard 2.0.0
pyyaml 5.3.1
tensorboard 2.2.1
torchvision 0.5.0
tqdm 4.50.2
```
## Setup
1. Set up a virtualenv with python 3.7.7 with conda.
2. Install the required packages.
3. Create a data directory as a base for all datasets, e.g., ./data/ in the code directory/
## Starting an Experiment 
```bash
python main.py --config <path/to/config> <override-args>
```
Common example ```override-args``` include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs, and ```--prune-rate``` to set the prune rate. Run ```python main --help``` for more details.
## Example Run
```bash
python main.py --config configs/resnet32-cifar100-pr0.1.yaml --multigpu 0 --data dataset/ --prune-rate 0.1
```