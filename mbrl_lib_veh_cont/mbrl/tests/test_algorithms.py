import os
import pathlib
import random
import tempfile

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.macura as macura
import mbrl.algorithms.m2ac as m2ac
import mbrl.env as mbrl_env

_TRIAL_LEN = 32
_NUM_TRIALS_MBPO = 12
_REW_C = 0.001
_INITIAL_EXPLORE = 500
_CONF_DIR = pathlib.Path("mbrl") / "examples" / "conf"

# Not optimal, but the prob. of observing this by random seems to be < 1e-5
_TARGET_REWARD = -20 * _REW_C

_REPO_DIR = pathlib.Path(os.getcwd())
_DIR = tempfile.TemporaryDirectory()

_SILENT = True
_DEBUG_MODE = False

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class MockLineEnv(gym.Env):
    def __init__(self):
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(2), np.inf * np.ones(2), shape=(2,)
        )
        self.action_space = gym.spaces.Box(-np.ones(1), np.ones(1), shape=(1,))
        self.action_space.seed(SEED)
        self.observation_space.seed(SEED)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        return np.array([self.pos, self.vel]), {}

    def step(self, action: np.ndarray):
        self.vel += action.item()
        self.pos += self.vel
        self.time_left -= 1
        reward = -_REW_C * (self.pos ** 2)
        return np.array([self.pos, self.vel]), reward, self.time_left == 0, False, {}


def mock_reward_fn(action, obs):
    return -_REW_C * (obs[:, 0] ** 2).unsqueeze(1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def test_mbpo():
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "mbpo.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / "gaussian_mlp_ensemble.yaml",
        "r",
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "num_steps": _NUM_TRIALS_MBPO * _TRIAL_LEN,
            "term_fn": "no_termination",
            "epoch_length": _TRIAL_LEN,
            "freq_train_model": _TRIAL_LEN // 4,
            "patience": 5,
            "model_lr": 1e-3,
            "model_wd": 5e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.2,
            "effective_model_rollouts_per_step": 400,
            "rollout_schedule": [1, _NUM_TRIALS_MBPO, 15, 15],
            "num_sac_updates_per_step": 40,
            "num_epochs_to_retain_sac_buffer": 1,
            "num_elites": 5,
            "sac_updates_every_steps": 1,
            "sac_gamma": 0.99,
            "sac_tau": 0.005,
            "sac_alpha": 0.2,
            "sac_policy": "Gaussian",
            "sac_target_update_interval": 4,
            "sac_automatic_entropy_tuning": True,
            "sac_hidden_size": 200,
            "sac_lr": 0.0003,
            "sac_batch_size": 256,
            "sac_target_entropy": -0.05,
            "exploration_type_env": "det",
            "model_hidden_size": 400,
            "minimum_variance_exponent":-10,
            "real_data_ratio":0.05
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 200,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.dynamics_model.ensemble_size = 7
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_MBPO + _INITIAL_EXPLORE

    env = MockLineEnv()
    test_env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = mbpo.train(
        env, test_env, term_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )
    assert max_reward > _TARGET_REWARD

def test_m2ac():
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "m2ac.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / "gaussian_mlp_ensemble.yaml",
        "r",
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "num_steps": _NUM_TRIALS_MBPO * _TRIAL_LEN,
            "term_fn": "no_termination",
            "epoch_length": _TRIAL_LEN,
            "freq_train_model": _TRIAL_LEN // 4,
            "patience": 5,
            "model_lr": 1e-3,
            "model_wd": 5e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.2,
            "effective_model_rollouts_per_step": 400,
            "num_sac_updates_per_step": 40,
            "num_epochs_to_retain_sac_buffer": 1,
            "num_elites": 5,
            "sac_updates_every_steps": 1,
            "sac_gamma": 0.99,
            "sac_tau": 0.005,
            "sac_alpha": 0.2,
            "sac_policy": "Gaussian",
            "sac_target_update_interval": 4,
            "sac_automatic_entropy_tuning": True,
            "sac_hidden_size": 200,
            "sac_lr": 0.0003,
            "sac_batch_size": 256,
            "sac_target_entropy": -0.05,
            "exploration_type_env": "det",
            "model_hidden_size": 400,
            "minimum_variance_exponent":-10,
            "real_data_ratio":0.05
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 200,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.dynamics_model.ensemble_size = 7
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_MBPO + _INITIAL_EXPLORE

    env = MockLineEnv()
    test_env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = m2ac.train(
        env, test_env, term_fn, cfg, silent=_SILENT, work_dir=None
    )
    assert max_reward > _TARGET_REWARD

def test_macura():
    with open(_REPO_DIR / _CONF_DIR / "algorithm" / "macura.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    with open(
        _REPO_DIR / _CONF_DIR / "dynamics_model" / "gaussian_mlp_ensemble.yaml",
        "r",
    ) as f:
        model_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": model_cfg,
        "overrides": {
            "num_steps": _NUM_TRIALS_MBPO * _TRIAL_LEN,
            "term_fn": "no_termination",
            "epoch_length": _TRIAL_LEN,
            "freq_train_model": _TRIAL_LEN // 4,
            "patience": 5,
            "model_lr": 1e-3,
            "model_wd": 5e-5,
            "model_batch_size": 256,
            "validation_ratio": 0.2,
            "effective_model_rollouts_per_step": 400,
            "num_sac_updates_per_step": 40,
            "num_epochs_to_retain_sac_buffer": 1,
            "num_elites": 5,
            "sac_updates_every_steps": 1,
            "sac_gamma": 0.99,
            "sac_tau": 0.005,
            "sac_alpha": 0.2,
            "sac_policy": "Gaussian",
            "sac_target_update_interval": 4,
            "sac_automatic_entropy_tuning": True,
            "sac_hidden_size": 200,
            "sac_lr": 0.0003,
            "sac_batch_size": 256,
            "sac_target_entropy": -0.05,
            "exploration_type_env": "det",
            "model_hidden_size": 400,
            "minimum_variance_exponent":-10,
            "real_data_ratio":0.05,
            "max_rollout_length": 10,
            "unc_tresh_run_avg_history":2000,
            "pink_noise_exploration_mod": False,
            "xi": 2.0,
            "zeta": 95,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 200,
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg.dynamics_model.ensemble_size = 7
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_MBPO + _INITIAL_EXPLORE

    env = MockLineEnv()
    test_env = MockLineEnv()
    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = macura.train(
        env, test_env,test_env, term_fn, cfg, silent=_SILENT, work_dir=None
    )
    assert max_reward > _TARGET_REWARD