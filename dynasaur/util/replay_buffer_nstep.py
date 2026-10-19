# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations
from collections import deque
import collections
import torch
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    effective_n_steps: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.randint(0, upper_bound, size=(batch_size,))
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBufferNStep(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    timeouts: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        gamma: float = 0.99,
        num_steps: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gamma = gamma
        self.num_steps = num_steps
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        # Convert numpy dtype to torch dtype
        obs_dtype = self._numpy_to_torch_dtype(observation_space.dtype)
        action_dtype = self._numpy_to_torch_dtype(self._maybe_cast_dtype(action_space.dtype))

        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_dtype, device=self.device)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_dtype, device=self.device)

        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_dtype, device=self.device)

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.effective_n_steps = th.ones((self.buffer_size, self.n_envs), dtype=th.int, device=self.device)
        self.terminations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.truncations = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.terminations.nbytes + self.truncations.nbytes + self.effective_n_steps.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        terminated: th.Tensor,
        truncated: th.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        # Ensure tensors are on the correct device
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)
        # Reshape needed when using multiple envs with discrete observations
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.clone()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs.clone()
        else:
            self.next_observations[self.pos] = next_obs.clone()

        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward.clone()

        self.terminations[self.pos] = terminated.clone()
        self.truncations[self.pos] = truncated.clone()
        self.effective_n_steps[self.pos] = 1
        #print(self.terminations[self.pos,0], self.truncations[self.pos,0], self.effective_n_steps[self.pos,0], self.rewards[self.pos,0], self.next_observations[self.pos,0,0])
        #print(self.terminations[self.pos+1-5:self.pos+1,0], self.truncations[self.pos+1-5:self.pos+1,0], self.effective_n_steps[self.pos+1-5:self.pos+1,0], self.rewards[self.pos+1-5:self.pos+1,0], self.next_observations[self.pos+1-5:self.pos+1,0,0])
        if self.num_steps > 1:
            done_target = torch.zeros_like(terminated, dtype=torch.bool, device=self.device)
            if self.full == False:#Edge case in beginning of buffer
                back_index = min(self.num_steps, self.pos+1)
            else:
                back_index = self.num_steps
            for num_step in range(1, back_index):
                #Two edge cases, if pos-self.num_steps <0 and if done is True
                target_index = (self.pos-num_step)%self.buffer_size
                done_target = torch.logical_or(torch.logical_or(self.terminations[target_index], self.truncations[target_index]), done_target)
                self.rewards[target_index] += (self.gamma**num_step * self.rewards[self.pos] ) * (~done_target).float()
                self.effective_n_steps[target_index] = torch.clamp(self.effective_n_steps[target_index]+(1*(~done_target).float()), max= self.num_steps)
                self.next_observations[target_index] = torch.where((~done_target).unsqueeze(1), self.next_observations[self.pos], self.next_observations[target_index])
                self.terminations[target_index] = torch.where((~done_target), self.terminations[self.pos], self.terminations[target_index])
        self.pos += 1
        #print(self.terminations[self.pos-5:self.pos,0], self.truncations[self.pos-5:self.pos,0], self.effective_n_steps[self.pos-5:self.pos,0], self.rewards[self.pos-5:self.pos,0], self.next_observations[self.pos-5:self.pos,0,0])
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def add_batch(
        self,
        obs_batch: th.Tensor,
        next_obs_batch: th.Tensor,
        action_batch: th.Tensor,
        reward_batch: th.Tensor,
        done_batch: th.Tensor,
        infos_batch: list[dict[str, Any]],
    ) -> None:
        # Ensure tensors are on the correct device
        num_steps = self.num_steps
        obs_batch = obs_batch.to(self.device)
        next_obs_batch = next_obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        batch_len = obs_batch.shape[0]
        action_batch = action_batch.reshape((batch_len, self.n_envs, self.action_dim))
        obs_batch = obs_batch.reshape((batch_len, self.n_envs, *self.obs_shape))
        next_obs_batch = next_obs_batch.reshape((batch_len, self.n_envs, *self.obs_shape))
        # Calculate num_steps discounted reward

        discount_factors = th.pow(self.gamma, th.arange(num_steps, device=self.device))
        # Expand reward_batch with num_steps zeros for boundary handling
        expanded_rewards = th.cat([reward_batch, th.zeros(num_steps, dtype=th.float32, device=self.device)])
        
        # Calculate num_steps discounted return for each position using vectorized operations
        discounted_rewards = th.zeros(batch_len, dtype=th.float32, device=self.device)
        
        for step in range(num_steps):
            discounted_rewards += expanded_rewards[step:batch_len + step] * discount_factors[step]
        next_obs_batch_copy = next_obs_batch.clone()
        done_batch_copy = done_batch.clone()
        
        if num_steps>1:
            if batch_len >= num_steps:
                next_obs_batch[:batch_len+1-num_steps] = next_obs_batch_copy[num_steps-1:]
                done_batch[:batch_len+1-num_steps] = done_batch_copy[num_steps-1:]
            for step in range(num_steps-1,0,-1):
                if batch_len-1-step>=0:
                    next_obs_batch[batch_len-1 - step] = next_obs_batch_copy[batch_len-1]
                    done_batch[batch_len-1 - step] = done_batch_copy[batch_len-1]


        reward_batch = discounted_rewards
        effective_n_steps = th.clamp(th.arange(batch_len,0, step=-1), max = num_steps).to(self.device)

        reward_batch = reward_batch.reshape((batch_len, self.n_envs))
        effective_n_steps = effective_n_steps.reshape((batch_len, self.n_envs))
        done_batch = done_batch.reshape((batch_len, self.n_envs))
        if self.pos + batch_len >= self.buffer_size:
            first_part_len = self.buffer_size - self.pos
            first_part_end = self.pos + first_part_len
            second_part_len = batch_len - first_part_len
            self.observations[self.pos:first_part_end] = obs_batch[:first_part_len].clone()
            self.next_observations[self.pos:first_part_end] = next_obs_batch[:first_part_len].clone()
            self.actions[self.pos:first_part_end] = action_batch[:first_part_len].clone()
            self.rewards[self.pos:first_part_end] = reward_batch[:first_part_len].clone()
            self.terminations[self.pos:first_part_end] = done_batch[:first_part_len].clone()
            self.effective_n_steps[self.pos:first_part_end] = effective_n_steps[:first_part_len].clone()

            self.observations[:second_part_len] = obs_batch[first_part_len:].clone()
            self.next_observations[:second_part_len] = next_obs_batch[first_part_len:].clone()
            self.actions[:second_part_len] = action_batch[first_part_len:].clone()
            self.rewards[:second_part_len] = reward_batch[first_part_len:].clone()
            self.terminations[:second_part_len] = done_batch[first_part_len:].clone()
            self.effective_n_steps[:second_part_len] = effective_n_steps[first_part_len:].clone()
            self.pos = second_part_len
            self.full = True
        else:
            end = self.pos + batch_len
            self.observations[self.pos:end] = obs_batch.clone()
            self.next_observations[self.pos:end] = next_obs_batch.clone()
            self.actions[self.pos:end] = action_batch.clone()
            self.rewards[self.pos:end] = reward_batch.clone()
            self.terminations[self.pos:end] = done_batch.clone()
            self.effective_n_steps[self.pos:end] = effective_n_steps.clone()
            self.pos += batch_len

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = th.randint(0, upper_bound, size=(batch_size,), device=self.device)
            return self._get_samples(batch_inds)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (th.randint(1, self.buffer_size, size=(batch_size,), device=self.device) + self.pos) % self.buffer_size
        else:
            batch_inds = th.randint(0, self.pos, size=(batch_size,), device=self.device)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples:
        # Sample randomly the env idx using PyTorch
        env_indices = th.randint(0, self.n_envs, size=(len(batch_inds),), device=self.device)

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.terminations[batch_inds, env_indices],
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
            self.effective_n_steps[batch_inds, env_indices].reshape(-1, 1)
        )
        return ReplayBufferSamples(*data)

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

    @staticmethod
    def _numpy_to_torch_dtype(dtype: np.typing.DTypeLike) -> th.dtype:
        """
        Convert numpy dtype to torch dtype.

        :param dtype: The numpy dtype
        :return: The corresponding torch dtype
        """
        if dtype == np.float64:
            return th.float64
        elif dtype == np.float32:
            return th.float32
        elif dtype == np.int64:
            return th.int64
        elif dtype == np.int32:
            return th.int32
        elif dtype == np.bool_:
            return th.bool
        elif dtype == np.uint8:
            return th.uint8
        else:
            return th.float32  # Default fallback