import torch
import numpy as np
from dynasaur.planning.sac_wrapper import SACAgent

class FilteredAgent:

    def __init__(self, perf_agent_fast_td_3_agent, safety_filter_fast_td3_agent, device, safety_env):
        self.perf_agent_fast_td_3_agent = perf_agent_fast_td_3_agent
        self.safety_filter_fast_td3_agent = safety_filter_fast_td3_agent
        self.device = device
        self.safety_wrapper_functions = safety_env

    def _act(self, obs: torch.Tensor
    ) -> np.ndarray:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        action_proposal = self.perf_agent_fast_td_3_agent.act(obs)
        if len(action_proposal.shape) == 0:
            action_proposal = action_proposal.unsqueeze(0)
        if len(action_proposal.shape) == 1:
            action_proposal = action_proposal.unsqueeze(1)
        action_hyperplane = self.safety_filter_fast_td3_agent.get_target_actor(obs)
        action_filterd, _ = self.safety_wrapper_functions.filter_action_batch_penalty(action_hyperplane, action_proposal)
        if action_filterd.shape[0]==1:
            action_filterd = action_filterd.squeeze(0)
        return action_filterd
    def act(
        self, obs: torch.Tensor
    ) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (torch.Tensor): the observation (or batch of observations) for which the action
                is needed.

        Returns:
            (np.ndarray): the filtered action.
        """
        action_filterd = self._act(obs)
        action_filterd = action_filterd.detach().cpu().numpy()
        return action_filterd
    
    def act_eps_penalty(
        self, obs: torch.Tensor, eps_explore: torch.Tensor=None,  noise_scale: torch.Tensor=None, random_explore=False
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Issues an action given an observation with exploration.

        Args:
            obs (torch.Tensor): the observation (or batch of observations) for which the action
                is needed.
            eps_explore (torch.Tensor, optional): exploration noise to add to the action.
            noise_scale (torch.Tensor, optional): scale factor for exploration noise.
            random_explore (bool): if True, uses eps_explore directly as action proposal.
                Defaults to False.

        Returns:
            tuple: A tuple containing:
                - (np.ndarray): the filtered action.
                - (torch.Tensor): the performance agent's proposed action.
                - (torch.Tensor): the penalty applied by the safety filter.
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if random_explore == False:
            action_proposal = self.perf_agent_fast_td_3_agent.explore(obs, eps_explore, noise_scale)
        else:
            # Directly take pink noise as explorative proposal actions
            action_proposal = eps_explore / 3

        action_hyperplane = self.safety_filter_fast_td3_agent.get_target_actor(obs)
        if len(action_proposal.shape) == 0:
            action_proposal = action_proposal.unsqueeze(0)
        if len(action_proposal.shape) == 1:
            action_proposal = action_proposal.unsqueeze(0)
        action_filterd, penalty = self.safety_wrapper_functions.filter_action_batch_penalty(action_hyperplane, action_proposal)
        perf_action = action_proposal
        if action_filterd.shape[0]==1:
            action_filterd = action_filterd.squeeze(0)
        return action_filterd, perf_action, penalty
    
    def explore(self, obs: torch.Tensor, eps_action: torch.Tensor=None,  noise_scales: torch.Tensor=None, random_explore=False)->torch.Tensor:
        return torch.clamp(self._act(obs) + noise_scales*eps_action,-1,1)