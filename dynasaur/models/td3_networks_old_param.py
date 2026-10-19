import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import NamedTuple
from dynasaur.util.data_management import save_data_compressed, load_data_compressed
import os

def sp_module(target, new_module, shrink_p, perturb_p):
    for target_param, new_param in zip(target.parameters(), new_module.parameters()):
        target_param.data.mul_(shrink_p).add_(perturb_p * new_param.data)

class QNetworkOldParam(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.fc1 = nn.Linear(
            np.array(observation_space.shape).prod() + np.prod(action_space.shape)+1,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def shrink_perturb(self, shrink_p=0.5, perturb_p=0.5):
        new_critic = QNetwork(self.observation_space, self.action_space, self.device).to(self.device)
        sp_module(self.fc1, new_critic.fc1, shrink_p, perturb_p)
        sp_module(self.fc2, new_critic.fc2, shrink_p, perturb_p)
        sp_module(self.fc3, new_critic.fc3, shrink_p, perturb_p)
class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.fc1 = nn.Linear(
            np.array(observation_space.shape).prod() + np.prod(action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def shrink_perturb(self, shrink_p=0.5, perturb_p=0.5):
        new_critic = QNetwork(self.observation_space, self.action_space, self.device).to(self.device)
        sp_module(self.fc1, new_critic.fc1, shrink_p, perturb_p)
        sp_module(self.fc2, new_critic.fc2, shrink_p, perturb_p)
        sp_module(self.fc3, new_critic.fc3, shrink_p, perturb_p)

class  Actor(nn.Module):
    def __init__(self, observation_space, action_space, device):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high[0] - action_space.low[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high[0] + action_space.low[0]) / 2.0,
                dtype=torch.float32,
            ),
        )
    def shrink_perturb(self, shrink_p=0.5, perturb_p=0.5):
        new_actor = Actor(self.env,self.observation_space, self.action_space, self.device).to(self.device)
        sp_module(self.fc1, new_actor.fc1, shrink_p, perturb_p)
        sp_module(self.fc2, new_actor.fc2, shrink_p, perturb_p)
        sp_module(self.fc_mu, new_actor.fc_mu, shrink_p, perturb_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class  ActorOldParam(nn.Module):
    def __init__(self, observation_space, action_space, device):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_space.shape)+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_mu(x)
        return x

    
class TD3Networks(NamedTuple):
    actor: Actor
    qf1: QNetwork
    qf2: QNetwork
    qf1_target: QNetwork
    qf2_target: QNetwork
    target_actor: Actor
    q_optimizer: optim.Adam
    actor_optimizer: optim.Adam
    
    def save(self,folder,name):
        """Save networks and optimizer states"""
        save_dict = {
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'qf1_target': self.qf1_target.state_dict(),
            'qf2_target': self.qf2_target.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
        }
        os.makedirs(folder, exist_ok=True)
        torch.save(save_dict, folder+'/'+name)
    
    @classmethod
    def load(cls, path, observation_space, action_space_control, args, device):
        """Load networks and recreate optimizers"""
        # Create fresh networks
        actor = Actor(observation_space,action_space_control, device).to(device)
        qf1 = QNetwork(observation_space, action_space_control).to(device)
        qf2 = QNetwork(observation_space, action_space_control).to(device)
        qf1_target = QNetwork(observation_space, action_space_control).to(device)
        qf2_target = QNetwork(observation_space, action_space_control).to(device)
        target_actor = Actor(observation_space, action_space_control, device).to(device)
        
        # Create fresh optimizers
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate_critic)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate_actor)
        
        # Load saved states
        checkpoint = torch.load(path)
        actor.load_state_dict(checkpoint['actor'])
        qf1.load_state_dict(checkpoint['qf1'])
        qf2.load_state_dict(checkpoint['qf2'])
        qf1_target.load_state_dict(checkpoint['qf1_target'])
        qf2_target.load_state_dict(checkpoint['qf2_target'])
        target_actor.load_state_dict(checkpoint['target_actor'])
        q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        
        return cls(actor, qf1, qf2, qf1_target, qf2_target, target_actor, q_optimizer, actor_optimizer)