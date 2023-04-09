import wandb
from tqdm import tqdm
try:
    from configs import max_target_returns, train_config
    from utils import make_dir, seed_everything, parse_json_dataset
    from dataset import ReplayBuffer
    from redq_bc import RandomizedEnsembles_BC
except ModuleNotFoundError:
    from .configs import max_target_returns, train_config
    from .utils import make_dir, seed_everything, parse_json_dataset
    from .dataset import ReplayBuffer
    from .redq_bc import RandomizedEnsembles_BC


def train_policy(max_returns=max_target_returns,
                 cfg=train_config()):
    
    max_return = 1.0
    if cfg.normalize_returns:
        max_return = max_returns[cfg.env]
    
    if cfg.save_model:
        make_dir("adaptive_bc/weights")
    
    seed_everything(cfg.seed)

    state_dim, action_dim, max_action = parse_json_dataset(cfg.env + '.json')
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.from_json(cfg.env + ".json")
    
    policy = RandomizedEnsembles_BC(state_dim,
                                    action_dim,
                                    max_action,
                                    cfg.discount_factor,
                                    cfg.tau,
                                    cfg.exploration_noise,
                                    cfg.noise_clip,
                                    cfg.policy_frequency,
                                    10,
                                    cfg.alpha_finetune,
                                    minimize_over_q=cfg.minimize_over_q,
                                    Kp=cfg.Kp,
                                    Kd=cfg.Kd)
    
    # with wandb.init(project="adaptive_bc", group=cfg.env, job_type="offline_train", name=f"redq_bc_{cfg.env}_{cfg.seed}"):
    #     wandb.config.update({k: v for k, v in cfg.__dict__.items() if not k.startswith('__')})
        
    for i in tqdm(range(cfg.pretrain_timesteps)):
        batch = replay_buffer.sample(cfg.batch_size)
        #print(batch[0].shape)
        pretrain_info = policy.train(batch)
            
        #wandb.log({"pretrain_info/": pretrain_info})
        
    if cfg.save_model:
        policy.save(f"adaptive_bc/{cfg.env}/redq_bc_{cfg.env}_{cfg.seed}")


if __name__ == "__main__":
    train_policy()
