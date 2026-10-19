import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import dynasaur.models as models
import dynasaur.planning as planning
import torch
from .replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    TransitionIterator,
)
from dynasaur.types_dynasaur import TransitionBatch
import dynasaur.util as util
from dynasaur.util.pink_noise import PinkNoiseGenerator

import imageio

def create_one_dim_tr_model(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
) -> models.OneDTransitionRewardModel:
    """Creates a 1-D transition reward model from a given configuration.

    This method creates a new model from the given configuration and wraps it into a
    :class:`models.OneDTransitionRewardModel` (see its documentation for explanation of some
    of the config args under ``cfg.algorithm``).
    The configuration should be structured as follows::

        -cfg
          -dynamics_model
            -model
              -_target_ (str): model Python class
              -in_size (int, optional): input size
              -out_size (int, optional): output size
              -model_arg_1
               ...
              -model_arg_n
          -algorithm
            -learned_rewards (bool): whether rewards should be learned or not
            -target_is_delta (bool): to be passed to the dynamics model wrapper
            -normalize (bool): to be passed to the dynamics model wrapper
          -overrides
            -no_delta_list (list[int], optional): to be passed to the dynamics model wrapper
            -obs_process_fn (str, optional): a Python function to pre-process observations
            -num_elites (int, optional): number of elite members for ensembles

    If ``cfg.dynamics_model.in_size`` is not provided, it will be automatically set to
    `obs_shape[0] + act_shape[0]`. If ``cfg.dynamics_model.out_size`` is not provided,
    it will be automatically set to `obs_shape[0] + int(cfg.algorithm.learned_rewards)`.

    The model will be instantiated using :func:`hydra.utils.instantiate` function.

    Args:
        cfg (omegaconf.DictConfig): the configuration to read.
        obs_shape (tuple of ints): the shape of the observations (only used if the model
            input or output sizes are not provided in the configuration).
        act_shape (tuple of ints): the shape of the actions (only used if the model input
            is not provided in the configuration).
        model_dir (str or pathlib.Path): If provided, the model will attempt to load its
            weights and normalization information from "model_dir / model.pth" and
            "model_dir / env_stats.pickle", respectively.

    Returns:
        (:class:`models.OneDTransitionRewardModel`): the model created.

    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg
    model_cfg = cfg.dynamics_model
    # Do not really get it because in member_cfg input/output is still ???
    if model_cfg._target_ == "models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    # set in_size to observation plus action dimensions
    if model_cfg.get("in_size", None) is None:
        model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    # set out size to observation + 1 for reward dimension
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    model = hydra.utils.instantiate(cfg.dynamics_model)

    # Possible to process each observation before putting it into the model as input together with the action
    name_obs_process_fn = cfg.algorithm.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.algorithm.obs_process_fn)
    else:
        obs_process_fn = None

    dynamics_model = models.OneDTransitionRewardModel(
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=cfg.algorithm.learned_rewards,
        obs_process_fn=obs_process_fn,
        no_delta_list=cfg.algorithm.get("no_delta_list", None),
        num_elites=cfg.algorithm.get("num_elites", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def load_hydra_cfg(results_dir: Union[str, pathlib.Path]) -> omegaconf.DictConfig:
    """Loads a Hydra configuration from the given directory path.

    Tries to load the configuration from "results_dir/.hydra/config.yaml".

    Args:
        results_dir (str or pathlib.Path): the path to the directory containing the config.

    Returns:
        (omegaconf.DictConfig): the loaded configuration.

    """
    results_dir = pathlib.Path(results_dir)
    cfg_file = results_dir / ".hydra" / "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")
    return cfg

def create_replay_buffer_fixed_capacity(
    cfg: omegaconf.DictConfig,
    capacity: int,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    obs_type: Type = np.float32,
    action_type: Type = np.float32,
    reward_type: Type = np.float32,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -num_steps (int, optional): how many steps to take in the environment
            -trial_length (int, optional): the maximum length for trials. Only needed if
                ``collect_trajectories == True``.

    The size of the replay buffer can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing ``cfg.algorithm.num_steps``.
    Specifying dataset set size directly takes precedence over number of steps.

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (Sequence of ints): the shape of observation arrays.
        act_shape (Sequence of ints): the shape of action arrays.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (:class:`replay_buffer.ReplayBuffer`): the replay buffer.
    """
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.algorithm.trial_length is None:
            raise ValueError(
                "cfg.algorithm.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len =cfg.algorithm.trial_length

    replay_buffer = ReplayBuffer(
        capacity,
        obs_shape,
        act_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer

def get_basic_buffer_iterators_different_distributions(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    bootstrap_permutes: bool = False,
    beginning_second_distribution=0
) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        ensemble_size (int): the size of the ensemble being trained.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
            the bootstrap data using permutations of the original data. Otherwise
            it will use sampling with replacement. Defaults to ``False``.

    Returns:
        (tuple of :class:`replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """
    data1 = replay_buffer.get_range_shuffled(start=0,end=beginning_second_distribution)
    val_size1 = int(beginning_second_distribution * val_ratio)
    train_size1 = beginning_second_distribution- val_size1
    train_data1 = data1[:train_size1]

    data2 = replay_buffer.get_range_shuffled(start=beginning_second_distribution,end=replay_buffer.num_stored)
    val_size2 = int((replay_buffer.num_stored-beginning_second_distribution) * val_ratio)
    train_size2 = (replay_buffer.num_stored-beginning_second_distribution)- val_size2
    train_data2 = data2[:train_size2]

    if train_size2>0:
        train_data = TransitionBatch(
            np.concatenate([train_data1.obs, train_data2.obs]),
            np.concatenate([train_data1.act, train_data2.act]),
            np.concatenate([train_data1.next_obs, train_data2.next_obs]),
            np.concatenate([train_data1.rewards, train_data2.rewards]),
            np.concatenate([train_data1.terminateds, train_data2.terminateds]),
            np.concatenate([train_data1.truncateds, train_data2.truncateds]),
        )
    else:
        train_data = train_data1

    val_size = val_size1+val_size2
    train_iter = BootstrapIterator(
        train_data,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=replay_buffer.rng,
    )

    val_iter = None
    if val_size > 0:
        val_data1 = data1[train_size1:]
        val_data2 = data2[train_size2:]
        if train_size2>0:
            val_data = TransitionBatch(
                np.concatenate([val_data1.obs, val_data2.obs]),
                np.concatenate([val_data1.act, val_data2.act]),
                np.concatenate([val_data1.next_obs, val_data2.next_obs]),
                np.concatenate([val_data1.rewards, val_data2.rewards]),
                np.concatenate([val_data1.terminateds, val_data2.terminateds]),
                np.concatenate([val_data1.truncateds, val_data2.truncateds]),
            )
        else:
            val_data = val_data1
        val_iter = TransitionIterator(
            val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
        )

    return train_iter, val_iter



def train_model_and_save_model_and_data(
    global_step: int,
    model: models.Model,
    model_trainer: models.ModelTrainer,
    args: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    work_dir: Optional[Union[str, pathlib.Path]] = None,
    callback: Optional[Callable] = None,
):
    """Convenience function for training a model and saving results.

    Runs `model_trainer.train()`, then saves the resulting model and the data used.
    If the model has an "update_normalizer" method it will be called before training,
    passing `replay_buffer.get_all()` as input.

    Args:
        model (:class:`models.Model`): the model to train.
        model_trainer (:class:`models.ModelTrainer`): the model trainer.
        cfg (:class:`omegaconf.DictConfig`): configuration to use for training. It
            must contain the following fields::

                -model_batch_size (int)
                -validation_ratio (float)
                -num_epochs_train_model (int, optional)
                -patience (int, optional)
                -bootstrap_permutes (bool, optional)
        replay_buffer (:class:`util.ReplayBuffer`): the replay buffer to use.
        work_dir (str or pathlib.Path, optional): if given, a directory to save
            model and buffer to.
        callback (callable, optional): if provided, this function will be called after
            every training epoch. See :class:`models.ModelTrainer` for signature.
    """
    # dataset_train is bootstrap iterator, so it gets seperate batches for each model to train.
    # dataset_val is a Transition iterator, because we only need to sample one batch out of the iterator
    # to validate if model ensemble is making good predicitons
    dataset_train, dataset_val = util.common.get_basic_buffer_iterators_different_distributions(
        replay_buffer,
        args.model_batch_size,
        args.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=args.get("bootstrap_permutes", False),
        beginning_second_distribution =args.prior_replaybuffer_size
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(replay_buffer.get_all())
    _,_,global_step = model_trainer.train(
        dataset_train,
        global_step,
        dataset_val=dataset_val,
        num_epochs=args.get("num_epochs_train_model", None),
        patience=args.get("patience", 1),
        improvement_threshold=args.get("improvement_threshold", 0.01),
        callback=callback,
    )
    if work_dir is not None:
        model.save(str(work_dir))
        replay_buffer.save(work_dir)
    return global_step

def gatherInitialData(
    env: gym.Env,
    steps_or_trials_to_collect: int,
    agent: planning.Agent,
    replay_buffer: Optional[ReplayBuffer],
    cfg ,
):
    step = 0
    pink_noise_exploration_generator = PinkNoiseGenerator(pink_noise_shape=(env.action_space.shape[0],), pink_noise_length=cfg.algorithm.trial_length, device=cfg.device,seed = cfg.seed)
    while True:
        obs, info= env.reset(options={"start_deviation":0.1})
        terminated = False
        truncated = False
        obs_cache = []
        action_cache = []
        next_obs_cache = []
        reward_cache = []
        terminated_cache = []
        truncated_cache = []
        if isinstance(agent, planning.Agent):
            agent.resetAgent()
        else:
            pink_noise_exploration_generator.reset_pink()
        frames=[]
        Return = 0
        while not terminated and not truncated:
            if isinstance(agent, planning.Agent):
                action = agent.act(obs, sample =True)
            else:
                action = agent.explore(torch.from_numpy(obs).float().cuda(), pink_noise_exploration_generator.get_pink_action(), noise_scales=torch.ones((1,), device=cfg.device, dtype=torch.float32)*cfg.algorithm.perf_args.exploration_noise_real_env_perf)
            action = action.detach().cpu().numpy().squeeze() if isinstance(action, torch.Tensor) else action
            action = env.action_space.sample() if step%3<=1 else action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            Return += reward
            frames.append(env.render())
            obs_cache.append(obs)
            next_obs_cache.append(next_obs)
            reward_cache.append(reward)
            action_cache.append(action)
            terminated_cache.append(terminated)
            truncated_cache.append(truncated)
            obs = next_obs
            step += 1
            if truncated and not terminated:
                print(Return, " reached in prior episode!")
                imageio.mimsave(f"prior_{replay_buffer.num_stored}.mp4", frames, fps=30)
                print("Initial Data: ", replay_buffer.num_stored, " steps collected so far of ", steps_or_trials_to_collect)
                for i in range(len(obs_cache)):
                    replay_buffer.add(obs_cache[i], action_cache[i], next_obs_cache[i], reward_cache[i], terminated_cache[i], truncated_cache[i])
                    if replay_buffer.num_stored >= steps_or_trials_to_collect:
                        print(f"Initial Data Collection - collected {steps_or_trials_to_collect} steps")
                        return
