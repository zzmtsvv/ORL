# Offline2Online Reinforcement Learning
This repository contains experiments of different reinforcement learning algorithms applied to 3 MuJoCo environments - `Walker2d, Hopper and Halfcheetah`. Essentially, there are 2 models in comparison: Adaptive Behavior Cloning Regularization [1] (in short, `redq_bc`) and Supported Policy Optimization for Offline Reinforcement Learning [2] (in short, `spot`).<br /><br />`July-August 2023 update`: There are also additional implementations of Cal-QL [9] in `cal_ql`, ReBRAC[11] in `rebrac`, EDAC[12] in `edac`, AWAC[13] in `awac`, Decision Transformer[14] in `decision_transformer`, IQL[15] in `iql` folders respectively. At the moment offline training is realised for this models. There is also a mention of my implementation of SAC-RND[10].

## General setup
I've chosen these datasets from gym as they are from MuJoCo, i.e. require learning of complex underlying structufe of the given task with trade-off in short-term and long-term strategies and Google Colab doesn't die from them ;). I have also used `d4rl` [3] library at https://github.com/tinkoff-ai/d4rl as a module to get offline dataset. Datasets used from `d4rl` for environments mentioned above: `medium` and `medium-replay`. Both models have the same base structure in architecture and training - actor-critic model [6] combined with Double Q-learning ([7], [8]).

Models (both redq_bc and spot) were trained on this offline dataset first using `Adam` optimizer with `lr = 3e-4`. The same with online training. Scripts can be found in appropriate folders (`adaptive_bc` and `spot`)

## Models

All available models can be tested in colab opening `inference.ipynb`. Examples of evaluation can be found in `video` folder.

### Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning
redq_bc is implemented to adaptively weigh the L2 loss associated with offline dataset distribution during online fine-tuning on order to stabilise the training. This loss is constructed into the architecture to prevent sudden distribution shift from offline to online data with such simple regularisation that requires minimum code changes (the method is located in the `adaptive_bc` folder, there is also `paper` folder with key moments from the following paper to realise the model). Logs are available at: https://wandb.ai/zzmtsvv/adaptive_bc


https://user-images.githubusercontent.com/85760987/230909607-2296499e-e0bf-4f7d-b1ae-f1947bd06bc7.mp4


### Supported Policy Optimization for Offline Reinforcement Learning
spot is also implemented to mitigate the problem of the distribution shift by adding a density-based constraint to the main objective. The offline behavior density is realised with Conditional VAE ([4], [5]) that reconstructs action joint with condition (state in this case). VAE is trained as usual and then its loss is used as a régularisation term in offline and online training (there is also additional cooling component in online fine-tuning for more stable handling of distribution shift). The method is located in the `spot` folder, there is also `paper` folder with key moments from the following paper to realise the model, Tensorboard plots can be seen in `graphs` folder.

https://user-images.githubusercontent.com/85760987/230911045-41823337-cc23-4c2f-9409-800739337310.mp4


## Results
As can be seen from plots and concrete examples on videos, `spot` performs much better than `redq_bc`. Intuitively, it can be connected with the fact both works brings additional regularization term during training, in fact, density-constraint support defined in spot can handle offline distribution support more succesfully than L2 term in redq_bc due to its bigger complexity. Furthermore, additional research on latent space of VAE can potentially bring impact in offline2online field.

# Offline Reinforcement Learning
## SAC-RND[10]
Check out for [my implementation](https://github.com/zzmtsvv/sac_rnd). Logs are available via the [link](https://wandb.ai/zzmtsvv/sac_rnd?workspace=user-zzmtsvv)

## Cal-QL
Although it is offline2online method, only offline training is realised yet in `cal_ql` folder. Logs are available via the [link](https://wandb.ai/zzmtsvv/cal_ql?workspace=user-zzmtsvv)

## ReBRAC
Logs are available via the [link](https://wandb.ai/zzmtsvv/ReBRAC?workspace=user-zzmtsvv)

## EDAC
- SAC-N: with `eta = 0`. [Logs](https://wandb.ai/zzmtsvv/SAC-N?workspace=user-zzmtsvv)
- LB-SAC: with `eta = 0` and `batch_size = 10_000`. [Logs](https://wandb.ai/zzmtsvv/LB-SAC?workspace=user-zzmtsvv)
- EDAC: [Logs](https://wandb.ai/zzmtsvv/EDAC?workspace=user-zzmtsvv)

## AWAC
Logs are available via the [link](https://wandb.ai/zzmtsvv/AWAC?workspace=user-zzmtsvv)

## Decision Transformer
Logs are available via the [link](https://wandb.ai/zzmtsvv/DecisionTransformer?workspace=user-zzmtsvv)

## IQL
Logs are available via the [link](https://wandb.ai/zzmtsvv/IQL?workspace=user-zzmtsvv)


## References
[1] - Yi Zhao et al. (2022). Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning. Available at: https://arxiv.org/abs/2210.13846 <br />
[2] - Jialong Wu et al. (2022). Supported Policy Optimization for Offline Reinforcement Learning. Available at: https://arxiv.org/abs/2202.06239 <br />
[3] - Justin Fu et al. (2021). D4RL: Datasets for Deep Data-driven Reinforcement Learning. Available at: https://arxiv.org/abs/2004.07219 <br />
[4] - Kingma, Welling et al. (2014). Auto-Encoding Variational Bayes. Available at: https://arxiv.org/abs/1312.6114 <br />
[5] - Sohn, Lee, Yan et al. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. Available at: https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html <br />
[6] - Lillicrap, Hunt et al. (2015). Continuous Control With Deep Reinforcement Learning. Available at: https://arxiv.org/abs/1509.02971 <br />
[7] - Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning. Available at: https://arxiv.org/abs/1312.5602 <br />
[8] - Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. Available at: https://arxiv.org/abs/1802.09477 <br />
[9] - Nakamoto, Zhai et al. (2023). Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning. Available at: https://arxiv.org/abs/2303.05479 <br />
[10] - Nikulin, Kurenkov et al. (2023). Anti-Exploration by Random Network Distillation. Available at: https://arxiv.org/abs/2301.13616 <br/>
[11] - Tarasov, Kurenkov et al. (2023). Revisiting the Minimalist Approach to Offline Reinforcement Learning. Available at: https://arxiv.org/abs/2305.09836 <br/>
[12] - An, Moon et al. (2021). Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble. Available at: https://arxiv.org/abs/2110.01548 <br/>
[13] - Nair, Gupta et al. (2021). AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. Available at: https://arxiv.org/abs/2006.09359 <br/>
[14] - Chen, Lu et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. Available at: https://arxiv.org/abs/2106.01345 <br/>
[15] - Kostrikov, Nair et al. (2021). Offline Reinforcement Learning with Implicit Q-Learning. Available at: https://arxiv.org/abs/2110.06169
