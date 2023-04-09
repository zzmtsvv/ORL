from dataclasses import dataclass


max_target_returns = {
    "halfcheetah-medium-replay-v0": 15.743,
    "halfcheetah-medium-v0": 15.743,
    "hopper-medium-replay-v0": 6.918,
    "hopper-medium-v0": 6.918,
    "walker2d-medium-replay-v0": 10.271,
    "walker2d-medium-v0": 10.271
}


@dataclass
class train_config:
    policy: str = "REDQ_BC"
    env: str = "hopper-medium-replay-v0" # [halfcheetah-medium-replay-v0 walker2d-medium-replay-v0]
    seed: int = 42
    eval_frequency: int = 5000
    max_timesteps: int = 250000
    pretrain_timesteps: int = 1000000
    num_updates: int = 10
    save_model: bool = True
    load_policy_path: str = ""
    episode_length: int = 1000
    exploration_noise: float = 0.1  # standard deviation of a gaussian devoted to the action space exploration noise
    batch_size: int = 256
    discount_factor: float = 0.99
    tau: float = 0.005  # see algo.jpeg in 'paper' folder
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    alpha: float = 0.4
    alpha_finetune: float = 0.4
    sample_method: str = "random"  # best
    sample_ratio: float = 0.05  # see algo.jpeg in 'paper' folder (ratio to keep offline data in replay buffer)
    minimize_over_q: bool = False  # if false, use randomized ensembles, else min Q values for steps, see eq3.PNG in 'paper' folder
    Kp: float = 0.00003 # see eq2.PNG in 'paper' folder
    Kd: float = 0.0001 # see eq2.PNG in 'paper' folder
    normalize_returns: bool = True  # if true, divide returns by a factor of a target return defined in 'max_target_returns' dataclass
    save_model: bool = True


if __name__ == "__main__":
    print({k: v for k, v in train_config.__dict__.items() if not k.startswith("__")})
     