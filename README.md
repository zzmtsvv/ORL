# Offline-to-Online Reinforcement Learning
This repository contains experiments of different reinforcement learning algorithms applied to 3 MuJoCo environments - `Walker2d, Hopper and Halfcheetah`. Essentially, there are 2 models in comparison: Adaptive Behavior Cloning Regularization [1] (in short, `redq_bc`) and Supported Policy Optimization for Offline Reinforcement Learning [2] (in short, `spot`).

## General setup
I've chosen these datasets from gym as they are from MuJoCo, i.e. require learning of complex underlying structufe of the given task with trade-off in short-term and long-term strategies and Google Colab doesn't die from them ;). I have also used `d4rl` [3] library at https://github.com/tinkoff-ai/d4rl. Datasets used from `d4rl` for environments mentioned above: `medium` and `medium-replay`. 



## References
[1] - Yi Zhao et al. (2022). Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning. Available at: https://arxiv.org/abs/2210.13846 <br />
[2] - Jialong Wu et al. (2022). Supported Policy Optimization for Offline Reinforcement Learning. Available at: https://arxiv.org/abs/2202.06239 <br />
[3] - Justin Fu et al. (2021). D4RL: DATASETS FOR DEEP DATA-DRIVEN REINFORCEMENT LEARNING. available at: https://arxiv.org/abs/2004.07219
