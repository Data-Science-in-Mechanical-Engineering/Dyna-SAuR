import torch
import torch.optim as optim
import wandb
from dynasaur.models.td3_networks import TD3Networks, QNetwork, Actor
import torch.nn.functional as F

class FastTD3Agent:
    def __init__(self, observation_space, action_space, device, cfg, old_networks = None, hyperplane_exploration = False, completeLoad: bool = False):
        self.device = device
        self.cfg = cfg
        # hyperplane_exploration=True → filter policy µ: S → U (eq. 13); False → control policy π: S → A.
        if hyperplane_exploration:
            self.args = cfg.algorithm.safety_args
        else:
            self.args = cfg.algorithm.perf_args
        self.one_dim_action_space = cfg.algorithm.env_safety_args.one_d_actionspace
        if self.args.normalize_observations:
            self.normalization_factors = torch.tensor(self.args.observation_normalization_factors, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.hyperplane_exploration = hyperplane_exploration
        self.train_iter_counter = 0
        self.observation_space = observation_space
        self.action_space = action_space
        self.actor = Actor(observation_space, action_space, device).to(device)
        self.qf1 = QNetwork(observation_space, action_space).to(device)
        self.qf2 = QNetwork(observation_space, action_space).to(device)
        self.qf1_target = QNetwork(observation_space, action_space).to(device)
        self.qf2_target = QNetwork(observation_space, action_space).to(device)
        self.target_actor = Actor(observation_space, action_space, device).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.learning_rate_critic)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.learning_rate_actor)
        if old_networks is not None: 
            if completeLoad:
                self.actor.load_state_dict(old_networks.actor.state_dict())
                self.actor_optimizer.load_state_dict(old_networks.actor_optimizer.state_dict())
                self.target_actor.load_state_dict(old_networks.target_actor.state_dict())
                self.qf1.load_state_dict(old_networks.qf1.state_dict())
                self.qf2.load_state_dict(old_networks.qf2.state_dict())
                self.qf1_target.load_state_dict(old_networks.qf1_target.state_dict())
                self.qf2_target.load_state_dict(old_networks.qf2_target.state_dict())
                self.q_optimizer.load_state_dict(old_networks.q_optimizer.state_dict())
        self.networks = TD3Networks(
            self.actor, self.qf1, self.qf2,
            self.qf1_target, self.qf2_target,
            self.target_actor,
            self.q_optimizer, self.actor_optimizer
        )
    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation if needed.

        Args:
            obs (torch.Tensor): the observation to normalize.
        """
        if self.args.normalize_observations:
           obs = obs * self.normalization_factors
        return obs
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Issues an action given an observation.

        Args:
            obs (torch.Tensor): the observation for which the action is needed.

        Returns:
            (torch.Tensor): the action.
        """
        with torch.no_grad():
            return self.get_actor(obs)
        
    def explore_hyperplane(self, obs: torch.Tensor,eps_action: torch.Tensor,noise_scales: torch.Tensor) -> torch.Tensor:
        """Issues an action given an observation with exploration noise.

        Args:
            obs (torch.Tensor): the observation for which the action is needed.

        Returns:
            (torch.Tensor): the action.
        """
        with torch.no_grad():
            action = self.act(obs.to(torch.float32))
            if len(action.shape)==1:
                action = action.unsqueeze(0)
            if len(noise_scales.shape)==1:
                noise_scales= noise_scales.unsqueeze(1) # because of broadcasting in action space
            if len(eps_action.shape) ==1:
                eps_action = eps_action.unsqueeze(0)
            # Add σ1·ϵ^µ pink noise to u_t then re-project onto U by dividing norms > 1,
            # preserving small norms so near-permissive hyperplanes remain reachable (Algorithm 4).
            action = action + eps_action * noise_scales
            action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
            action = action / torch.clamp(torch.linalg.norm(action, dim=1, keepdim=True), min=1)
            return action

    def explore(self, obs: torch.Tensor, eps_action: torch.Tensor,noise_scales: torch.Tensor) -> torch.Tensor:
        """Issues an action given an observation.

        Args:
            obs (torch.Tensor): the observation for which the action is needed.

        Returns:
            (torch.Tensor): the action.
        """
        with torch.no_grad():
            action = self.act(obs.to(torch.float32))
            if len(action.shape)==1:
                action = action.unsqueeze(0)
            if len(noise_scales.shape)==1:
                noise_scales= noise_scales.unsqueeze(1) # because of broadcasting in action space
            if len(eps_action.shape) ==1:
                eps_action = eps_action.unsqueeze(0)
            action = action + eps_action * noise_scales
            action = action.clamp(self.action_space.low[0], self.action_space.high[0])
            return action


    def get_td3_networks(self):
        return self.networks
    
    def get_qf1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.qf1(self.normalize_observation(obs), action)
    def get_qf2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.qf2(self.normalize_observation(obs), action)
    def get_qf1_target(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.qf1_target(self.normalize_observation(obs), action)
    def get_qf2_target(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.qf2_target(self.normalize_observation(obs), action)
    def get_actor(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.actor(self.normalize_observation(obs))
        # For the filter policy µ: project output into U (unit hyperball, eq. 13) by dividing
        # norms > 1, keeping norms < 1 intact so the actor can output near-permissive filters.
        # Skipped for 1-D action spaces where U reduces to a scalar norm in (-1, 1).
        if self.hyperplane_exploration and not self.one_dim_action_space:
            norm = torch.norm(action, dim=1, keepdim=True)
            action = action /  torch.clamp(norm, min=1)
        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        return action
    def get_target_actor(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.target_actor(self.normalize_observation(obs))
        if self.hyperplane_exploration and not self.one_dim_action_space:
            norm = torch.norm(action, dim=1, keepdim=True)
            action = action/ torch.clamp(norm, min=1)
        action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        return action
    
    def train(self, data, global_step, total_perf_env_steps):
        with torch.no_grad():
            current_next_state = data.next_observations.to(torch.float32)
            next_dones = data.dones
            next_state_actions = self.get_target_actor(current_next_state)
            # TD3 target policy smoothing: add clipped Gaussian noise to break exploitation
            # of Q-function errors at deterministic target actions (standard TD3 trick).
            noise = (torch.randn_like(next_state_actions) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip)
            next_state_actions = (next_state_actions + noise).clamp(self.action_space.low[0], self.action_space.high[0])
            # Re-project noisy target action back onto U so bootstrapped Q-values are consistent
            # with the filter policy's constrained output space (eq. 13).
            next_state_actions /= torch.clamp(torch.linalg.norm(next_state_actions, dim=1, keepdim=True), min=1)

            qf1_next_target = self.get_qf1_target(current_next_state, next_state_actions)
            qf2_next_target = self.get_qf2_target(current_next_state, next_state_actions)
            # Double Q clipped target reduces overestimation bias (standard TD3).
            min_qf_next_target = torch.clamp(torch.min(qf1_next_target, qf2_next_target), min=-100, max=100)
            # n-step return: γ^n discounting via effective_n_steps supports multi-step TD targets (FastTD3).
            next_q_value = (data.rewards).view(-1) + (1 - next_dones.to(torch.float32)).view(-1) * (self.args.gamma**data.effective_n_steps.view(-1)) * (min_qf_next_target).view(-1)
        observation_batch = data.observations.to(torch.float32)
        qf1_a_values = self.get_qf1(observation_batch, data.actions).view(-1)
        qf2_a_values = self.get_qf2(observation_batch, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value, reduction='none')
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value, reduction='none')
        qf1_loss = (qf1_loss).mean()
        qf2_loss = (qf2_loss).mean()
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        # Delayed policy update: actor and targets updated every policy_frequency critic steps (TD3).
        if self.train_iter_counter % self.args.policy_frequency == 0:
            actor_loss = -self.get_qf1(observation_batch, self.get_actor(observation_batch)).mean()

            # "Freedom" regulariser for µ: maximize Q^SF − c·||u_t||_2 (eq. 18).
            # The ||u_t||_2 penalty discourages unnecessarily restrictive filters by penalising
            # large norms, which correspond to tighter halfspace constraints via h (eq. 14).
            if self.hyperplane_exploration and self.cfg.algorithm.get("meta_loss", True):
                actor_loss+=self.cfg.algorithm.safety_args.meta_loss_factor*torch.norm(self.get_actor(observation_batch),dim=1).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Polyak averaging for actor and both critic target networks (standard TD3).
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        prefix = "Safety_Filter_Learning-TrainingLosses" if self.hyperplane_exploration else "PerformanceAgentTraining-Losses"
        if self.train_iter_counter % self.cfg.algorithm.log_interval == 0:
            if self.cfg.track_wandb:
                wandb.log({
                    f"{prefix}/qf1_max_values": qf1_a_values.max().item(),
                    f"{prefix}/qf2_max_values": qf2_a_values.max().item(),
                    f"{prefix}/qf1_min_values": qf1_a_values.min().item(),
                    f"{prefix}/qf2_min_values": qf2_a_values.min().item(),
                    f"{prefix}/qf1_values": qf1_a_values.mean().item(),
                    f"{prefix}/qf2_values": qf2_a_values.mean().item(),
                    f"{prefix}/qf_target_values_max": next_q_value.max().item(),
                    f"{prefix}/qf_target_values_min": next_q_value.min().item(),
                    f"{prefix}/qf_target_values": next_q_value.mean().item(),
                    f"{prefix}/qf1_loss": qf1_loss.item(),
                    f"{prefix}/qf2_loss": qf2_loss.item(),
                    f"{prefix}/qf_loss": qf_loss.item() / 2.0,
                    f"{prefix}/actor_loss": actor_loss.item(),
                    f"{prefix}/mean_batch_reward_training": (data.rewards).view(-1).mean().item(),
                    f"{prefix}/max_batch_reward_training": (data.rewards).view(-1).max().item(),
                    f"{prefix}/min_batch_reward_training": (data.rewards).view(-1).min().item(),
                    f"{prefix}/total_steps": total_perf_env_steps,
                }, step=global_step)
        self.train_iter_counter+=1 