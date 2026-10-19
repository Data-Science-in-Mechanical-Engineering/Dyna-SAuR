import hydra
import numpy as np
import omegaconf
import torch
import random
from dynasaur.env.cartpole_continuous import CartPoleEnv
from dynasaur.algorithms.dynasaur import train as dynasaur
from dynasaur.algorithms.dynasaur_old_param import train as dynasaur_old_param
from dynasaur.env.walker_custom import Walker2dEnvCustom
from gymnasium.wrappers import TimeLimit
from dynasaur.util.termination_and_reward_functions import cartpole_termination, cartpole_reward, cartpole_starting_states, walker2d_termination, walker2d_reward, walker_starting_states, walker2d_termination_model, cartpole_termination_model
@hydra.main(config_path="conf", config_name="main", version_base="1.1")
def run(cfg: omegaconf.DictConfig):
    algo_name = cfg.algorithm.name
    print("Running "+algo_name+ " algorithm")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if cfg.algorithm.name == "dynasaur_cartpole_old_param":
            env = CartPoleEnv(render_mode="rgb_array", screen_width=1080, screen_height=1080)
            test_env =  CartPoleEnv(render_mode="rgb_array", screen_width=1080, screen_height=1080)
            return dynasaur_old_param(cfg, env, test_env,cartpole_termination, cartpole_reward,cartpole_starting_states, cartpole_termination_model)
    if cfg.algorithm.name == "dynasaur_walker_old_param":
            env = Walker2dEnvCustom(render_mode="rgb_array", width = 1080, height = 1080)
            env = TimeLimit(env, max_episode_steps=cfg.algorithm.trial_length)
            test_env =  Walker2dEnvCustom(render_mode="rgb_array", width = 1080, height = 1080)
            test_env = TimeLimit(test_env, max_episode_steps=cfg.algorithm.trial_length)
            dynasaur_old_param(cfg, env, test_env,walker2d_termination, walker2d_reward, walker_starting_states, walker2d_termination_model)
    if cfg.algorithm.name == "dynasaur_cartpole":
        env = CartPoleEnv(render_mode="rgb_array", screen_width=1200, screen_height=800)
        test_env =  CartPoleEnv(render_mode="rgb_array", screen_width=1200, screen_height=800)
        return dynasaur(cfg, env, test_env,cartpole_termination, cartpole_reward,cartpole_starting_states, cartpole_termination_model)
    if cfg.algorithm.name == "dynasaur_walker":
        env = Walker2dEnvCustom(render_mode="rgb_array", width = 1080, height = 1080)
        env = TimeLimit(env, max_episode_steps=cfg.algorithm.trial_length)
        test_env =  Walker2dEnvCustom(render_mode="rgb_array", width = 1080, height = 1080)
        test_env = TimeLimit(test_env, max_episode_steps=cfg.algorithm.trial_length)
        return dynasaur(cfg, env, test_env,walker2d_termination, walker2d_reward, walker_starting_states, walker2d_termination_model)

if __name__ == "__main__":
        run()
