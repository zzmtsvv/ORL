# Offline & Offline2Online Reinforcement Learning
`April 2023`: This repository contains experiments of different reinforcement learning algorithms applied to 3 MuJoCo environments - `Walker2d, Hopper and Halfcheetah`. Essentially, there are 2 models in comparison: Adaptive Behavior Cloning Regularization [1] (in short, `redq_bc`) and Supported Policy Optimization for Offline Reinforcement Learning [2] (in short, `spot`).<br /><br />
`July-September 2023 update`: There are also additional implementations of:

- Cal-QL [9]: [Logs](https://wandb.ai/zzmtsvv/cal_ql?workspace=user-zzmtsvv)
- ReBRAC[11]: [Logs](https://wandb.ai/zzmtsvv/ReBRAC?workspace=user-zzmtsvv)
- EDAC[12]: Logs: [EDAC itself](https://wandb.ai/zzmtsvv/EDAC?workspace=user-zzmtsvv), [SAC-N[12]](https://wandb.ai/zzmtsvv/SAC-N?workspace=user-zzmtsvv) (with `eta = 0`), [LB-SAC[16]](https://wandb.ai/zzmtsvv/LB-SAC?workspace=user-zzmtsvv) (with `eta = 0` and `batch_size = 10_000`)
- AWAC[13]: [Logs](https://wandb.ai/zzmtsvv/AWAC?workspace=user-zzmtsvv)
- Decision Transformer[14]: [Logs](https://wandb.ai/zzmtsvv/DecisionTransformer?workspace=user-zzmtsvv)
- IQL[15]: [Logs](https://wandb.ai/zzmtsvv/IQL?workspace=user-zzmtsvv)
- MSG[17]: [Logs](https://wandb.ai/zzmtsvv/MSG?workspace=user-zzmtsvv) (This method is realised upon offline SAC-N algorithm. However, my realization lacks appropriate hyperparameters for best results.)
- PRDC[19]: [Logs](https://wandb.ai/zzmtsvv/PRDC?workspace=user-zzmtsvv)
- DOGE[20]: [Logs](https://wandb.ai/zzmtsvv/DOGE?workspace=user-zzmtsvv)
- BEAR[21]: [Logs](https://wandb.ai/zzmtsvv/BEAR?workspace=user-zzmtsvv)
- SAC-RND[10]: [Logs](https://wandb.ai/zzmtsvv/sac_rnd?workspace=user-zzmtsvv) & [Implementation](https://github.com/zzmtsvv/sac_rnd)
- RORL: [Logs](https://wandb.ai/zzmtsvv/RORL?workspace=user-zzmtsvv) & [Implementation](https://github.com/zzmtsvv/rorl) (lacks appropriate hyperparameters)
- CNF[18]: [Logs](https://wandb.ai/zzmtsvv/CNF/workspace?workspace=user-zzmtsvv) & [Implementation](https://github.com/zzmtsvv/cnf)
- offline O3F[22]: [Logs](https://wandb.ai/zzmtsvv/offline_O3F?workspace=user-zzmtsvv) (realised for offline learning, not as stated in the paper)
- XQL[23]: [Logs](https://wandb.ai/zzmtsvv/XQL?workspace=user-zzmtsvv)
- TD7[24]: [Logs](https://wandb.ai/zzmtsvv/TD7?workspace=user-zzmtsvv)
- offline TQC[25]: [Logs](https://wandb.ai/zzmtsvv/offline_TQC?workspace=user-zzmtsvv) (failed on `walker2d-medium-v2`)
- InAC[26]: [Logs](https://wandb.ai/zzmtsvv/InAC?workspace=user-zzmtsvv)
- FisherBRC[27]: [Logs](https://wandb.ai/zzmtsvv/FisherBRC?workspace=user-zzmtsvv)
- Diffusion Q-Learning[28]: [Logs](https://wandb.ai/zzmtsvv/DiffusionQL?workspace=user-zzmtsvv)
- Sparse Q-Learning[29]: [Logs](https://wandb.ai/zzmtsvv/SQL?workspace=user-zzmtsvv)
- Exponential Q-Learning[29]: [Logs](https://wandb.ai/zzmtsvv/EQL?workspace=user-zzmtsvv) (differs from SQL mentioned above by a bit different update of value function and actor)

At the moment offline training is realised for this models. Logs (of only training actually, unfortunately, without evaluation as it was forbidden on the machine to install mujoco stuff, so I trained the models with preloaded pickle and json datasets) are available up below.

You can also check out my [Logs](https://wandb.ai/zzmtsvv/SEEM?workspace=user-zzmtsvv) for SEEM[30] paper.

## General setup (April 2023)
I've chosen these datasets from gym as they are from MuJoCo, i.e. require learning of complex underlying structufe of the given task with trade-off in short-term and long-term strategies and Google Colab doesn't die from them ;). I have also used `d4rl` [3] library at https://github.com/tinkoff-ai/d4rl as a module to get offline dataset. Datasets used from `d4rl` for environments mentioned above: `medium` and `medium-replay`. Both models have the same base structure in architecture and training - actor-critic model [6] combined with Double Q-learning ([7], [8]).

Models (both redq_bc and spot) were trained on this offline dataset first using `Adam` optimizer with `lr = 3e-4`. The same with online training. Scripts can be found in appropriate folders (`adaptive_bc` and `spot`)

## Models (April 2023)

All available models can be tested in colab opening `inference.ipynb`. Examples of evaluation can be found in `video` folder.

### Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning
redq_bc is implemented to adaptively weigh the L2 loss associated with offline dataset distribution during online fine-tuning on order to stabilise the training. This loss is constructed into the architecture to prevent sudden distribution shift from offline to online data with such simple regularisation that requires minimum code changes (the method is located in the `adaptive_bc` folder, there is also `paper` folder with key moments from the following paper to realise the model). Logs are available at: https://wandb.ai/zzmtsvv/adaptive_bc


https://user-images.githubusercontent.com/85760987/230909607-2296499e-e0bf-4f7d-b1ae-f1947bd06bc7.mp4


### Supported Policy Optimization for Offline Reinforcement Learning
spot is also implemented to mitigate the problem of the distribution shift by adding a density-based constraint to the main objective. The offline behavior density is realised with Conditional VAE ([4], [5]) that reconstructs action joint with condition (state in this case). VAE is trained as usual and then its loss is used as a régularisation term in offline and online training (there is also additional cooling component in online fine-tuning for more stable handling of distribution shift). The method is located in the `spot` folder, there is also `paper` folder with key moments from the following paper to realise the model, Tensorboard plots can be seen in `graphs` folder.

https://user-images.githubusercontent.com/85760987/230911045-41823337-cc23-4c2f-9409-800739337310.mp4


## Results (April 2023)
As can be seen from plots and concrete examples on videos, `spot` performs much better than `redq_bc`. Intuitively, it can be connected with the fact both works brings additional regularization term during training, in fact, density-constraint support defined in spot can handle offline distribution support more succesfully than L2 term in redq_bc due to its bigger complexity. Furthermore, additional research on latent space of VAE can potentially bring impact in offline2online field.


## References
[1] - Yi Zhao et al. (2022). [Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning](https://arxiv.org/abs/2210.13846). <br/>
[2] - Jialong Wu et al. (2022). [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/abs/2202.06239). <br />
[3] - Justin Fu et al. (2021). [D4RL: Datasets for Deep Data-driven Reinforcement Learning](https://arxiv.org/abs/2004.07219). <br />
[4] - Kingma, Welling et al. (2014). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). <br />
[5] - Sohn, Lee, Yan et al. (2015). [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html). <br />
[6] - Lillicrap, Hunt et al. (2015). [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971). <br />
[7] - Mnih et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). <br />
[8] - Fujimoto et al. (2018). [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). <br />
[9] - Nakamoto, Zhai et al. (2023). [Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479). <br />
[10] - Nikulin, Kurenkov et al. (2023). [Anti-Exploration by Random Network Distillation](https://arxiv.org/abs/2301.13616). <br/>
[11] - Tarasov, Kurenkov et al. (2023). [Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2305.09836). <br/>
[12] - An, Moon et al. (2021). [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548).<br/>
[13] - Nair, Gupta et al. (2021). [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359).<br/>
[14] - Chen, Lu et al. (2021). [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).<br/>
[15] - Kostrikov, Nair et al. (2021). [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).<br/>
[16] - Nikulin, Kurenkov et al. (2022). [Q-Ensemble for Offline RL: Don't Scale the Ensemble, Scale the Batch Size](https://arxiv.org/abs/2211.11092).<br/>
[17] - Kamyar, Ghasemipour et al. (2022). [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703). <br/>
[18] Akimov, Kurenkov et al. (2023). [Let Offline RL Flow: Training Conservative Agents in the Latent Space of Normalizing Flows](https://arxiv.org/abs/2211.11096).<br/>
[19] Ran, Li et al. (2023). [Policy Regularization with Dataset Constraint for Offline Reinforcement Learning](https://arxiv.org/abs/2306.06569). <br/>
[20] Li, Zhan et al. (2023). [When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning](https://arxiv.org/abs/2205.11027).<br/>
[21] Kumar, Fu et al. (2019). [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://proceedings.neurips.cc/paper_files/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf).<br/>
[22] Mark, Ghardizadeh et al. (2023). [Fine-Tuning Offline Policies With Optimistic Action Selection](https://openreview.net/forum?id=2x8EKbGU51k). <br/>
[23] Garg, Hejna et al. (2023). [Extreme Q-Learning: MaxEnt RL without Entropy](https://arxiv.org/abs/2301.02328) <br/>
[24] Fujimoto, Chang et al. (2023). [For SALE: State-Action Representation Learning for Deep Reinforcement Learning](https://arxiv.org/abs/2306.02451) <br/>
[25] Kuznetsov, Shvechikov et al. (2020). [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269) <br/>
[26] Xiao, Wang et al. (2023). [The In-Sample Softmax for Offline Reinforcement Learning](https://arxiv.org/abs/2302.14372) <br/>
[27] Kostrikov, Tompson, et al. (2021). [Offline Reinforcement Learning with Fisher Divergence Critic Regularization](https://arxiv.org/abs/2103.08050) <br/>
[28] Wang, Hunt, Zhou (2023). [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193) <br/>
[29] Xu, Jiang et al. (2023). [Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization](https://arxiv.org/abs/2303.15810v1) <br/>
[30] Yue, Lu et al. (2023). [Understanding, Predicting and Better Resolving Q-Value Divergence in Offline-RL](https://arxiv.org/abs/2310.04411)
