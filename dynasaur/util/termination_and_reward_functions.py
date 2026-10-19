import torch
import numpy as np
import math
#Cart Pole Functions
def cartpole_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 4.8
    theta_threshold_radians = 24 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done
def cartpole_termination_model(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 4.6
    theta_threshold_radians = 22 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done

def cartpole_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    terminated = cartpole_termination(act, next_obs).float().view(-1)
    return -10*(terminated) + (1.0-terminated)*(torch.exp(-0.2* torch.pow((next_obs[:,0]-4.1),2)))

def cartpole_starting_states(batch_size):
    low, high = (-0.05,0.05)
    start_state =  np.random.uniform(low=low, high=high, size=(batch_size,4))
    return start_state

# Walker Functions
def walker_starting_states(batch_size):
    start_state = np.tile(np.array([1, -0.5,  0,-1.2,  0.5,  0, -1.2,0.5,0,0,0,0,0,0,0,0,0]), (batch_size, 1))
    low, high = (-5e-3,5e-3)
    start_state += np.random.uniform(low=low, high=high, size=(batch_size,17))
    return start_state

def walker2d_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.8) * (height < 1.2) * (angle > -0.9) * (angle < 0.9)
    done = ~not_done
    done = done[:, None]
    return done
def walker2d_termination_model(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = (height > 0.81) * (height < 1.19) * (angle > -0.89) * (angle < 0.89)
    done = ~not_done
    done = done[:, None]
    return done
def walker2d_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2
    x_velocity = next_obs[:, 8]
    not_healty = walker2d_termination(act, next_obs).float().view(-1)
    control_cost = 1e-3 * torch.sum(torch.square(act), dim=1)
    reward = x_velocity - control_cost - (not_healty*10)
    return reward

