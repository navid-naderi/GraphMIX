# GraphMIX: Graph Convolutional Value Decomposition in Multi-Agent Reinforcement Learning

GraphMIX is a multi-agent deep reinforcement learning (MARL) approach that relies on a graph neural network (GNN) architecture for combining the individual agent value functions into a global team value funtion, and it provides state-of-the-art results across several maps in the StarCraft Multi-Agent Challenge ([SMAC](https://github.com/oxwhirl/smac)) benchmark.

This repository contains the PyTorch-based implementation of GraphMIX and relies on the [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) libraries.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

Use the following command to run GraphMIX on any desired SMAC map (e.g., `corridor` in the example below):

```shell
python3 src/main.py --config=graphmix --env-config=sc2 with \
env_args.map_name=corridor lambda_local=1 test_interval=20000 test_nepisode=32 \
t_max=5000000 epsilon_anneal_time=500000
```

The parameter `lambda_local` represents the local loss coefficient in the GraphMIX objective function.

All results will be stored in the `results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

