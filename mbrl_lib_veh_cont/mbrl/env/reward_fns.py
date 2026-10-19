# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns

class linearized_simple_pendulum:
    def __init__(self, Q, R, device):
        self.Q = torch.FloatTensor(Q).reshape((2,2)).to(device)
        self.R = torch.FloatTensor(R).to(device)

    def __call__(self, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        assert len(next_obs.shape) == len(act.shape) == 2
        reward = (next_obs[:,None,:] @ self.Q[None,:,:] @ next_obs[:,:, None]).squeeze(-1) + self.R @ torch.square(act)
        return reward.view(-1, 1)


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    terminated = termination_fns.cartpole(act, next_obs).float().view(-1)
    return -10*(terminated) + (1.0-terminated)*(torch.exp(-0.2* torch.pow((next_obs[:,0]-4.1),2)))


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)

def custom_walker2d_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2
    x_velocity = next_obs[:, 8]
    not_healty = termination_fns.custom_walker2d_termination(act, next_obs).float().view(-1)
    control_cost = 1e-3 * torch.sum(torch.square(act), dim=1)
    reward = x_velocity - control_cost - (not_healty*10)
    return reward

def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)
