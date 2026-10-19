from gymnasium import  spaces
import gymnasium
import torch
import numpy as np
from dynasaur.util.pink_noise import PinkNoiseGenerator
class HyperplaneSafetyWrapperEnv(gymnasium.Wrapper):
    def double_to_hyperplane(self, action):
        # Bijection h: U → (w, b) from Theorem 5.1 (eq. 14).
        # w = u / ||u||_2  (unit normal vector pointing in the direction of u)
        # b = (2||u||_2 - 1) * ||w||_1  (offset scaled so ||u||_2 near 0 → permissive, near 1 → restrictive)
        if isinstance(action, torch.Tensor):
            norm = torch.norm(action).clip(min=1e-6,max=1-1e-3)
            direction_h = action/norm
            max_offset = torch.sum(torch.abs(direction_h))
            b_h = (2 * norm - 1)*max_offset
            return direction_h, b_h
        else:
            norm = np.linalg.norm(action).clip(min=1e-6,max=1-1e-3)
            direction_h = action/norm
            max_offset = np.sum(np.abs(direction_h))
            b_h = (2 * norm - 1)*max_offset
            return direction_h, b_h

    def double_to_hyperplane_batch(self, action_batch):
        # Batched version of h (eq. 14): maps a batch of u_t ∈ U to (w_t, b_t) hyperplane parameters.
        if isinstance(action_batch, torch.Tensor):
            norm = torch.norm(action_batch, dim=1, keepdim=True).clip(min=1e-6,max=1-1e-3)
            direction_h = action_batch/norm
            max_offset = torch.sum(torch.abs(direction_h), dim=1, keepdim=True)
            b_h = (2 * norm - 1)*max_offset
            return direction_h, b_h
        elif isinstance(action_batch, np.ndarray):
            norm = np.linalg.norm(action_batch, axis=1, keepdims=True).clip(min=1e-6,max=1-1e-3)
            direction_h = action_batch/norm
            max_offset = np.sum(np.abs(direction_h), axis=1, keepdims=True)
            b_h = (2 * norm - 1)*max_offset
            return direction_h, b_h

    def __init__(self, base_env: gymnasium.Env, cfg, eval_mode = False):
        super().__init__(base_env)
        assert len(self.env.action_space.shape) == 1
        self.epsilon = 1e-8
        self._update_running_mean = True
        self.device = cfg.device
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.max_loss_threshold = self.env.max_loss_threshold
        self.parallel_models = self.cfg.algorithm.safety_args.parallel_models
        self.starting_states = []
        # Each parallel rollout is randomly assigned either the perf-policy or pure pink-noise 
        self.pink_and_not_perf = torch.where(torch.rand((self.parallel_models,), device=self.device) < 0.5, torch.zeros((self.parallel_models,), device=self.device, dtype=bool), torch.ones((self.parallel_models,), device=self.device, dtype=bool))
        # σ1 ~ U(σ_min, σ_max): per-rollout exploration noise scale, re-sampled each episode (Algorithm 4).
        self.noise_scales = torch.rand((self.parallel_models,), device=self.device)*(self.cfg.algorithm.safety_args.max_noise_scale - self.cfg.algorithm.safety_args.min_noise_scale) + self.cfg.algorithm.safety_args.min_noise_scale
        self.original_low = torch.tensor(self.env.action_space.low).to(self.device)
        self.original_high = torch.tensor(self.env.action_space.high).to(self.device)
        self.action_space_control = self.env.action_space
        self.action_space_dim = self.action_space_control.shape[0]
        
        #PinkNoiseActionGenerator
        batch_size = self.parallel_models if not self.eval_mode else self.cfg.algorithm.safety_args.eval_episodes*2
        self.pink_noise_actions_rollout_generator = PinkNoiseGenerator(pink_noise_shape=(batch_size,self.action_space_dim), pink_noise_length=self.cfg.algorithm.trial_length, device=self.device, seed = cfg.seed)
        
        act_high = torch.full((self.parallel_models, self.action_space_dim,), 1, dtype=torch.float32).to(self.device)
        self.action_space = spaces.Box(-act_high.cpu().numpy(), act_high.cpu().numpy(), dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.episode_length = 0
        self.episode_return = 0
        
        self.x_threshold = self.env.x_threshold if hasattr(self.env, 'x_threshold') else None
        self.theta_threshold_radians = self.env.theta_threshold_radians if hasattr(self.env, 'theta_threshold_radians') else None
        self.width = self.env.width if hasattr(self.env, 'width') else None
        self.height = self.env.height if hasattr(self.env, 'height') else None
        self.wind_x_start = self.env.wind_x_start if hasattr(self.env, 'wind_x_start') else None
        self.wind_x_end = self.env.wind_x_end if hasattr(self.env, 'wind_x_end') else None
        self.wind_y_start = self.env.wind_y_start if hasattr(self.env, 'wind_y_start') else None
        self.wind_y_end = self.env.wind_y_end if hasattr(self.env, 'wind_y_end') else None
        self.termination_penalty = self.cfg.algorithm.safety_args.termiantion_penalty
        self.safe_episode_length = self.cfg.algorithm.trial_length
        self.training_episode_length = self.cfg.algorithm.safety_args.training_episode_length
    
    def _reward_safety_filter(self, action_hyperplane_batch, epistemic_frac):
        return 1
    
    def filter_action_batch_penalty(self, action_hyperplane_batch, action_proposal_batch):
        # Implements κ^π_µ (eq. 6/7): project a_t onto the safe halfspace w^T a ≥ b clipped to A.
        # The projection a^V_t = argmin ||a - a_t||_2 s.t. w^T a ≥ b is solved iteratively:
        orig_data_points = action_proposal_batch.clone()
        direction_h_batch , b_h_batch = self.double_to_hyperplane_batch(action_hyperplane_batch)
        action_proposal_batch = orig_data_points.clone()
        for _ in range (self.action_space_dim):
            dot_product = torch.sum(direction_h_batch * action_proposal_batch, dim=1, keepdim=True)
            mask = (dot_product) < b_h_batch
            adjustment = ((dot_product - b_h_batch) ) * direction_h_batch
            action_filtered = torch.where(mask, action_proposal_batch - adjustment, action_proposal_batch)
            action_filtered = action_filtered.clip(-1,1)
            first_point = action_filtered
            action_proposal_batch = action_filtered
            dot_product = torch.sum(direction_h_batch * action_proposal_batch, dim=1, keepdim=True)
            mask = (dot_product) < b_h_batch
            adjustment = ((dot_product - b_h_batch) ) * direction_h_batch
            action_filtered = torch.where(mask, action_proposal_batch - adjustment, action_proposal_batch)
            action_filtered = action_filtered.clip(-1,1)
            second_point = action_filtered
            # Exact solve for actions still oscillating between the two gradient steps.
            indices_converged = torch.norm(first_point - second_point, dim=1) < 1e-12
            if (~indices_converged).any():
                t_solve =  (b_h_batch[~indices_converged]-torch.sum(first_point[~indices_converged]*direction_h_batch[~indices_converged], axis=1, keepdims=True))  / ( torch.sum((first_point[~indices_converged]-second_point[~indices_converged])*direction_h_batch[~indices_converged], axis=1, keepdims=True))
                action_filtered[~indices_converged] = first_point[~indices_converged] + (first_point[~indices_converged]-second_point[~indices_converged])*t_solve
            action_proposal_batch = action_filtered
        assert torch.all(torch.sum(direction_h_batch * action_filtered, dim=1, keepdim=True) >= b_h_batch - 1e-4), "Action filtering failed to project all actions onto safe side of hyperplane."
        assert torch.all(action_filtered <= 1.0+ 1e-4) and torch.all(action_filtered >= -1.0 - 1e-4), "Action filtering produced actions outside of valid range."
        action_filtered = action_filtered.clip(-1,1)
        return  self.scale_action(action_filtered), torch.norm(orig_data_points-action_filtered, dim=1, keepdim=True)
    
    def step_batch(self,state_batch, action_hyperplane_batch, action_proposal_perf_batch = None, noPink = False):
        if not noPink:
            action_proposal_batch = self.pink_noise_actions_rollout_generator.get_pink_action()/3
        else:
            action_proposal_batch = None
        if action_proposal_perf_batch is not None:
            action_proposal_perf_batch = action_proposal_perf_batch.reshape(-1,self.action_space_dim)
            if self.eval_mode==False:
                action_proposal_batch = torch.where(self.pink_and_not_perf.unsqueeze(-1),action_proposal_perf_batch, action_proposal_batch)
            else:
                if action_proposal_batch is None:
                    action_proposal_batch = action_proposal_perf_batch
                else:
                    action_proposal_batch[:action_proposal_perf_batch.shape[0],:] = action_proposal_perf_batch
        action_batch_orig_scale, _ = self.filter_action_batch_penalty(action_hyperplane_batch, action_proposal_batch)
        next_state, pred_dones, too_much_single_stop_uncertainty, added_infoloss, control_reward, epistemic_frac = self.env.step_batch(state_batch, action_batch_orig_scale)

        # Terminate if the state enters S_F (pred_dones) or the model leaves E via λ1 (eq. 23).
        terminated = torch.logical_or(pred_dones, too_much_single_stop_uncertainty)

        # r^SF = 1 if (s, a^V) ∈ S_S(E), else −1/(1−γ^SF) applied in the buffer at episode end (eq. 15).
        reward = self._reward_safety_filter(action_hyperplane_batch, epistemic_frac)

        self.episode_return += reward
        self.episode_length += 1
        self.current_total_infoloss_batch += added_infoloss
        # Truncate when the horizon T is reached or accumulated info loss exceeds λ2 (eq. 25).
        truncated_length = torch.where((self.episode_length >= (self.safe_episode_length if self.eval_mode else self.training_episode_length)).unsqueeze(1), torch.ones_like(terminated, dtype=torch.bool), torch.zeros_like(terminated, dtype=torch.bool))
        truncated_infoloss = (self.current_total_infoloss_batch > (self.max_loss_threshold*self.safe_episode_length if self.eval_mode else self.max_loss_threshold*self.training_episode_length)).any(dim=1, keepdim=True)
        truncated = torch.logical_or(truncated_length, truncated_infoloss)

        return next_state, (~(terminated.squeeze()))*reward+((terminated.squeeze())*self.termination_penalty), terminated, truncated, control_reward

    def scale_action(self, action):
         scaled_action = self.original_low + (action + 1.0) * 0.5 * (self.original_high - self.original_low)
         return scaled_action

    def reset_batch(self, states_to_replace_maybe=None, done_tensor=None):
        batch_size = self.parallel_models if not self.eval_mode else self.cfg.algorithm.safety_args.eval_episodes*2
        initial_state_batch, info = self.env.reset_batch(eval_mode=self.eval_mode, batch_size = batch_size)
        # Re-sample behavior policy assignment and σ1 for each (newly) done rollout (Algorithm 4).
        pink_and_not_perf_new = torch.where(torch.rand((self.parallel_models,), device=self.device) < 0.5, torch.zeros((self.parallel_models,), device=self.device, dtype=bool), torch.ones((self.parallel_models,), device=self.device, dtype=bool))
        new_scales = torch.rand((self.parallel_models,), device=self.device)*(self.cfg.algorithm.safety_args.max_noise_scale - self.cfg.algorithm.safety_args.min_noise_scale) + self.cfg.algorithm.safety_args.min_noise_scale
        if states_to_replace_maybe is not None and done_tensor is not None:
            # Selective reset: only overwrite state and episode counters for finished rollouts,
            # preserving in-progress rollouts so the batch stays fully utilised (Algorithm 4).
            initial_state_batch = torch.where(done_tensor, initial_state_batch, states_to_replace_maybe)
            self.pink_and_not_perf = torch.where(done_tensor.squeeze(), pink_and_not_perf_new, self.pink_and_not_perf)
            self.noise_scales = torch.where(done_tensor.squeeze(), new_scales, self.noise_scales)
            self.episode_return = torch.where(done_tensor.squeeze(),torch.zeros((batch_size,)).to(self.device), self.episode_return)
            self.episode_length =torch.where(done_tensor.squeeze(), torch.zeros((batch_size)).to(self.device), self.episode_length)
            # Reset per-rollout accumulated info loss λ2 counter (eq. 25) for done episodes only.
            self.current_total_infoloss_batch = torch.where(done_tensor,torch.zeros((batch_size, self.observation_space.shape[0])).to(self.device), self.current_total_infoloss_batch)
        else:
            # Full reset at the start of a new filter training phase.
            self.episode_return = torch.zeros((batch_size,)).to(self.device)
            self.episode_length = torch.zeros((batch_size,)).to(self.device)
            self.current_total_infoloss_batch = torch.zeros((batch_size, self.observation_space.shape[0])).to(self.device)
            self.noise_scales = new_scales
            self.pink_and_not_perf = pink_and_not_perf_new

        return initial_state_batch, info

