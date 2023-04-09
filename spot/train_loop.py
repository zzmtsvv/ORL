import os
import torch
from torch.nn import functional as F
import numpy as np
import gym
#import d4rl
from tqdm import tqdm, trange

try:
    from configs import vae_config, spot_config
    from utils import make_dir, seed_everything, DummyScheduler, VideoRecorder, parse_json_dataset
    from spot_ import SPOT
    from vae import ConditionalVAE
    from dataset import ReplayBuffer
    from logger import Logger
except ModuleNotFoundError:
    from .configs import vae_config, spot_config
    from .utils import make_dir, seed_everything, DummyScheduler, VideoRecorder, parse_json_dataset
    from .spot_ import SPOT
    from .vae import ConditionalVAE
    from .dataset import ReplayBuffer
    from .logger import Logger



def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_vae(cfg: vae_config):
    make_dir(os.path.join(cfg.base_dir, cfg.weights_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    dir_ = os.path.join(cfg.base_dir, cfg.weights_dir)
    logger = Logger(os.path.join(cfg.base_dir, "runs"), use_tb=True)

    '''
    env = gym.make(f"{cfg.env}-{cfg.dataset}-{cfg.version}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    '''
    state_dim, action_dim, max_action = parse_json_dataset(cfg.env_name)


    if not cfg.max_action_exists:
        max_action = None
    
    latent_dim = action_dim * 2
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    #replay_buffer.from_d4rl(d4rl.qlearning_dataset(env))

    if cfg.normalize_states:
        mean, std = replay_buffer.normalize_states()
    
    if cfg.clip_to_eps:
        replay_buffer.clip()
    
    states = replay_buffer.states
    actions = replay_buffer.actions
    
    eval_states = None
    eval_actions = None
    if cfg.eval_size:
        eval_size = int(states.shape[0] * cfg.eval_size)
        eval_indexes = np.random.choice(states.shape[0], eval_size, replace=False)
        train_indexes = np.setdiff1d(np.arange(states.shape[0]), eval_indexes)

        eval_states = states[eval_indexes].to(device)
        eval_actions = actions[eval_indexes].to(device)

        states = states[train_indexes]
        actions = actions[train_indexes]
    
    vae = ConditionalVAE(state_dim, action_dim, latent_dim, max_action=max_action, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = DummyScheduler()

    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    
    total_size = states.shape[0]
    batch_size = cfg.batch_size

    for step in tqdm(range(1, cfg.num_iterations + 1), desc="train"):
        indexes = np.random.choice(total_size, batch_size)
        train_states = states[indexes].to(device)
        train_actions = actions[indexes].to(device)

        reconstructed, mean, std = vae(train_states, train_actions)
        
        recon_loss = F.mse_loss(reconstructed, train_actions)
        kl_loss = -1 / 2 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + cfg.beta * kl_loss

        logger.log('train/recon_loss', recon_loss, step=step)
        logger.log('train/kl_loss', kl_loss, step=step)
        logger.log('train/vae_loss', vae_loss, step=step)

        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()

        if not step % 5000:
            logger.dump(step)
            torch.save(vae.state_dict(), f"{dir_}/vae_{cfg.env}_{cfg.dataset}.pt")

            if eval_states is not None and eval_actions is not None:
                vae.eval()

                with torch.no_grad():
                    recon, mean, std = vae(eval_states, eval_actions)

                    recon_loss =  F.mse_loss(reconstructed, eval_actions)
                    kl_loss = -1 / 2 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                    vae_loss = recon_loss + cfg.beta * kl_loss

                    logger.log('train/recon_loss', recon_loss, step=step)
                    logger.log('train/kl_loss', kl_loss, step=step)
                    logger.log('train/vae_loss', vae_loss, step=step)
                vae.train()
        
        if cfg.use_scheduler and not (step + 1) % 10000:
            logger.log("train/lr", get_lr(optimizer), step=step)
            scheduler.step()
    
    logger._sw.close()


def eval_policy(cfg: spot_config,
                iteration: int,
                recorder: VideoRecorder,
                logger: Logger,
                policy: SPOT,
                env_name: str,
                seed: int,
                mean: np.ndarray,
                std: np.ndarray,
                eval_episodes: int = 10):
    env = gym.make(env_name)
    env.seed(seed + 100)

    lengths, returns, last_rewards = [], [], []
    average_reward = 0.0

    for episode in trange(eval_episodes):
        recorder.init(enabled=cfg.save_video)
        state, done = env.reset(), False
        
        #recorder.record(env)
        steps = 0
        episode_return = 0

        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.act(state)

            state, reward, done, _ = env.step(action)
            recorder.record(env)

            average_reward += reward
            episode_return += reward
            steps += 1

        lengths.append(steps)
        returns.append(episode_return)
        last_rewards.append(reward)
        recorder.save(f"evaluation_{iteration}_episode{episode}_return_{episode_return}.mp4")
    
    average_reward /= eval_episodes
    #d4rl_score = env.get_normalized_score(average_reward)

    if logger is not None:
        logger.log('eval/lengths_mean', np.mean(lengths), iteration)
        logger.log('eval/lengths_std', np.std(lengths), iteration)
        logger.log('eval/returns_mean', np.mean(returns), iteration)
        logger.log('eval/returns_std', np.std(returns), iteration)
        #logger.log('eval/d4rl_score', d4rl_score, iteration)
    
    #return d4rl_score


def train_policy(cfg=spot_config()):
    video_dir = os.path.join(cfg.base_dir, "video")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = os.path.join(cfg.base_dir, cfg.weights_dir)
    make_dir(weights_dir)
    make_dir(video_dir)

    #env = gym.make(cfg.env_name)

    seed_everything(cfg.seed)

    
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    
    state_dim, action_dim, max_action = parse_json_dataset(cfg.env_name)

    vae = ConditionalVAE(state_dim, action_dim, action_dim * 2, max_action).to(device)
    vae.load(cfg.vae_model_path)
    vae.eval()

    policy = SPOT(vae,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action,
                  discount_factor=cfg.discount_factor,
                  tau=cfg.tau,
                  policy_noise=cfg.policy_noise,
                  noise_clip=cfg.noise_clip,
                  policy_frequency=cfg.policy_frequency,
                  beta=cfg.beta,
                  lambda_=cfg.lambda_,
                  lr=cfg.lr,
                  actor_lr=cfg.actor_lr,
                  with_q_norm=cfg.with_q_norm,
                  num_samples=cfg.num_samples,
                  use_importance_sampling=cfg.use_importance_sampling,
                  actor_hidden_dim=cfg.actor_hidden_dim,
                  actor_dropout=cfg.actor_dropout)
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size=cfg.buffer_size)
    #replay_buffer.from_d4rl(d4rl.qlearning_dataset(env))
    replay_buffer.from_json(cfg.env_name)
    #assert replay_buffer.size + cfg.max_timesteps <= replay_buffer.buffer_size

    mean, std = 0, 1
    if cfg.normalize_env:
        mean, std = replay_buffer.normalize_states()
    
    if cfg.clip:
        replay_buffer.clip()
    
    logger = Logger(os.path.join(cfg.base_dir, "runs"), use_tb=True)
    recorder = VideoRecorder(video_dir)

    for timestep in trange(cfg.max_timesteps):
        policy.train(replay_buffer, batch_size=cfg.batch_size, logger=logger)

        # if not (timestep + 1) % cfg.eval_frequency:
        #     d4rl_score = eval_policy(cfg,
        #                              timestep + 1,
        #                              recorder,
        #                              logger,
        #                              policy,
        #                              cfg.env_name,
        #                              cfg.seed,
        #                              torch.from_numpy(mean),
        #                              torch.from_numpy(std),
        #                              cfg.eval_episodes)
            
        #     if cfg.save_model:
        #         policy.save(weights_dir)
    
    if cfg.save_final_model:
        policy.save(weights_dir)
    
    logger._sw.close()


if __name__ == "__main__":
    train_policy()
