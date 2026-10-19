import math
import random
from typing import Optional
import numpy as np
import torch
import gymnasium
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorednoise
from dynasaur.util.data_management import save_data_compressed, load_data_compressed
from dynasaur.util.plotting import  plot_starting_states_failure, plot_starting_states_reward, visualize_accurately_modelled_set

class dynasaurModelWrapper(gymnasium.Wrapper):
    def __init__(self, model_env, replay_buffer, cfg, env, device, starting_state_fn):
        self.starting_state_fn = starting_state_fn
        self.device = device
        self.cfg = cfg
        self.one_d_extra_visuals = cfg.algorithm.env_safety_args.one_d_extra_visuals
        if self.one_d_extra_visuals:
            self.theta_threshold_radians = env.theta_threshold_radians
            self.x_threshold = env.x_threshold
        self.observation_space = env.observation_space
        self.quantization_bits = torch.log2(torch.tensor(1e-6)).to(device)
        self.debug_mode = (cfg.get("debug_mode", False) and cfg.track_wandb)
        #a,b of a hyperplane
        self.action_space = env.action_space
        self.original_low = torch.tensor(env.action_space.low).to(device)
        self.original_high = torch.tensor(env.action_space.high).to(device)
        self.model_env = model_env
        self.env = env
        self.replay_buffer = replay_buffer
        self.starting_states_reward = None
        self.starting_states_failures = None
        self.starting_states_reward_model = np.empty((0, self.observation_space.shape[0]))
        self.ground_truth_starting_states_reward = np.empty((0, self.observation_space.shape[0]))
        self.starting_states_reward_model_max_reward = -float('inf')
        self.recalculate_threshold(replay_buffer=replay_buffer)
        print("Thresholds Kalmna Gain:",self.per_step_epistemic_frac)

    def get_real_next_state(self, state_batch, action_batch):
        next_state, terminated = self.env.step_batch_torch(state_batch, action_batch)
        return next_state, terminated
    
    def get_info_data_set_all(self):
        x_range = torch.arange(-2*self.x_threshold, 2*self.x_threshold, 0.2, device=self.device)  # First dimension: -2.4 to 2.4
        theta_range = torch.arange(-2*self.theta_threshold_radians, 2*self.theta_threshold_radians, math.pi / 360 *8,device=self.device)  # Third dimension: -2 to 2
        x_dot_range = torch.arange(-1, 1.01, 0.5, device=self.device)
        theta_dot_range = torch.arange(-1, 1.01, 0.5,device=self.device)
        # Create the grid for first and third dimensions
        x_grid, theta_grid, x_dot_grid, theta_dot_grid_ = torch.meshgrid(x_range, theta_range,x_dot_range,theta_dot_range,indexing='ij')

        obs_flat = torch.stack([
            x_grid.flatten(),  
            x_dot_grid.flatten(),  
            theta_grid.flatten(),  
            theta_dot_grid_.flatten()   
        ], dim=-1)

        actions_dataset1 = torch.rand((obs_flat.shape[0], 1), device=self.device) * 2 - 1  # uniform(-1, 1)
        actions_dataset2 = torch.rand((obs_flat.shape[0], 1), device=self.device) * 2 - 1
        actions_dataset3 = torch.rand((obs_flat.shape[0], 1), device=self.device) * 2 - 1
        actions_dataset4 = torch.rand((obs_flat.shape[0], 1), device=self.device) * 2 - 1
        actions_dataset5 = torch.rand((obs_flat.shape[0], 1), device=self.device) * 2 - 1

        model_state = {"obs": obs_flat}
        pred_next_obs1, pred_rewards, pred_dones, next_model_state1 = self.model_env.info_step(
            actions_dataset1, model_state, sample=True
        )
        real_next_states1, _ = self.get_real_next_state(obs_flat, actions_dataset1)
        difference_accuaracy_real1 = torch.sqrt(torch.sum(torch.pow(pred_next_obs1-real_next_states1,2),dim=1))
        difference_accuaracy_real1 = torch.sum((difference_accuaracy_real1/25).reshape(-1,25), axis=1)

        model_state = {"obs": obs_flat}
        pred_next_obs2, pred_rewards, pred_dones, next_model_state2 = self.model_env.info_step(
            actions_dataset2, model_state, sample=True
        )
        real_next_states2, _ = self.get_real_next_state(obs_flat, actions_dataset2)
        difference_accuaracy_real2 = torch.sqrt(torch.sum(torch.pow(pred_next_obs2-real_next_states2,2),dim=1))
        difference_accuaracy_real2 = torch.sum((difference_accuaracy_real2/25).reshape(-1,25), axis=1)


        model_state = {"obs": obs_flat}
        pred_next_obs3, pred_rewards, pred_dones, next_model_state3 = self.model_env.info_step(
            actions_dataset3, model_state, sample=True
        )
        real_next_states3, _ = self.get_real_next_state(obs_flat, actions_dataset3)
        difference_accuaracy_real3 = torch.sqrt(torch.sum(torch.pow(pred_next_obs3-real_next_states3,2),dim=1))
        difference_accuaracy_real3 = torch.sum((difference_accuaracy_real3/25).reshape(-1,25), axis=1)


        model_state = {"obs": obs_flat}
        pred_next_obs4, pred_rewards, pred_dones, next_model_state4 = self.model_env.info_step(
            actions_dataset4, model_state, sample=True
        )
        real_next_states4, _ = self.get_real_next_state(obs_flat, actions_dataset4)
        difference_accuaracy_real4 = torch.sqrt(torch.sum(torch.pow(pred_next_obs4-real_next_states4,2),dim=1))
        difference_accuaracy_real4 = torch.sum((difference_accuaracy_real4/25).reshape(-1,25), axis=1)

        model_state = {"obs": obs_flat}
        pred_next_obs5, pred_rewards, pred_dones, next_model_state5 = self.model_env.info_step(
            actions_dataset5, model_state, sample=True
        )
        real_next_states5, _ = self.get_real_next_state(obs_flat, actions_dataset5)
        difference_accuaracy_real5 = torch.sqrt(torch.sum(torch.pow(pred_next_obs5-real_next_states5,2),dim=1))
        difference_accuaracy_real5 = torch.sum((difference_accuaracy_real5/25).reshape(-1,25), axis=1)
        difference_accuaracy_real = (difference_accuaracy_real1 + difference_accuaracy_real2 + difference_accuaracy_real3 + difference_accuaracy_real4 + difference_accuaracy_real5)/5
        lost_K_dataset = (next_model_state1["K"] +next_model_state2["K"] +next_model_state3["K"]+next_model_state4["K"]+next_model_state5["K"])/5
        conditional_var = (next_model_state1["conditional_var"]+next_model_state2["conditional_var"]  +next_model_state3["conditional_var"]  +next_model_state4["conditional_var"]  +next_model_state5["conditional_var"] )/5    
        lost_info_dataset = torch.clamp(0.5*torch.log2(2*torch.pi*torch.exp(torch.tensor(1))*conditional_var) - self.quantization_bits, torch.tensor(0), torch.inf)
        return lost_K_dataset,lost_info_dataset, difference_accuaracy_real

    def visualize_accurate_set(self,index_dynasaur, global_step,total_perf_env_steps):
        lost_K_dataset, self.lost_info_dataset_all, difference_accuaracy_real = self.get_info_data_set_all()
        lost_K_dataset = lost_K_dataset.detach().cpu().numpy()
        self.lost_info_dataset_all = self.lost_info_dataset_all.detach().cpu().numpy()
        difference_accuaracy_real = difference_accuaracy_real.detach().cpu().numpy()
        wandb.log({"Model_Learning-thresholds/lost_info_dataset_all_x_max": self.lost_info_dataset_all.max(axis=0)[0],
                   "Model_Learning-thresholds/lost_info_dataset_all_xdot_max": self.lost_info_dataset_all.max(axis=0)[1],
                   "Model_Learning-thresholds/lost_info_dataset_all_theta_max": self.lost_info_dataset_all.max(axis=0)[2],
                   "Model_Learning-thresholds/lost_info_dataset_all_theta_dot_max": self.lost_info_dataset_all.max(axis=0)[3], 
                   "Model_Learning-thresholds/lost_1mk_dataset_replaybuffer_x_max": lost_K_dataset.max(axis=0)[0],
                   "Model_Learning-thresholds/lost_1mk_dataset_replaybuffer_xdot_max": lost_K_dataset.max(axis=0)[1],
                   "Model_Learning-thresholds/lost_1mk_dataset_replaybuffer_theta_max": lost_K_dataset.max(axis=0)[2],
                   "Model_Learning-thresholds/lost_1mk_dataset_replaybuffer_thetadot_max": lost_K_dataset.max(axis=0)[3],
                   }, step = global_step)
        
        
        x_range = np.arange(-2*self.x_threshold, 2*self.x_threshold, 0.2)  # First dimension: -2.4 to 2.4
        z_range = np.arange(-2*self.theta_threshold_radians, 2*self.theta_threshold_radians, math.pi / 360 *8)  # Third dimension: -2 to 2
        z_grid_plot,x_grid_plot = np.meshgrid(z_range, x_range)
        average_lost_K = np.sum((lost_K_dataset/25).reshape(-1,4,25), axis=2)
        average_lost_info = np.sum((self.lost_info_dataset_all/25).reshape(-1,4,25), axis=2)
        save_data_compressed(difference_accuaracy_real, f"Accuracy/{index_dynasaur}-accuracy")
        save_data_compressed(average_lost_K, f"LostInfo/{index_dynasaur}-k_average_lost_info")

        visualize_accurately_modelled_set(x_grid_plot, z_grid_plot, difference_accuaracy_real, average_lost_info, average_lost_K,self.per_step_epistemic_frac, global_step, self.cfg.algorithm.env_safety_args, total_perf_env_steps) if self.debug_mode else None
        
    def recalculate_threshold(self, replay_buffer):
        obs = replay_buffer.obs[:min(replay_buffer.num_stored,100000),:]
        self.obs_shape_model = obs.shape
        actions_dataset = replay_buffer.action[:min(replay_buffer.num_stored,100000),:]
        self.model_state = {"obs": obs}
        pred_next_obs, pred_rewards, pred_dones, next_model_state = self.model_env.info_step(
            actions_dataset, self.model_state, sample=True
        )
        conditional_var = next_model_state["conditional_var"]   
        lost_info_dataset = torch.clamp(0.5*torch.log2(2*torch.pi*torch.exp(torch.tensor(1))*conditional_var) - self.quantization_bits, torch.tensor(0), torch.inf)
        self.per_step_epistemic_infoloss = torch.quantile(lost_info_dataset, q=0.99, dim=0, keepdim=True).to(self.device)
        self.per_step_epistemic_frac = torch.quantile(next_model_state["K"], q=0.99, dim=0, keepdim=True).to(self.device) + self.cfg.algorithm.safety_args.dynamic_kalman_added_term
        print("Updated per-step lost info threshold:", self.per_step_epistemic_infoloss)
        print("Updated per-step kalman gain threshold:", self.per_step_epistemic_frac)
        if self.cfg.algorithm.safety_args.fixed_safety_kalman_threshold:
            self.per_step_epistemic_frac = torch.tensor(self.cfg.algorithm.safety_args.safety_kalman_threshold, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.max_loss_threshold = torch.quantile(lost_info_dataset, q=0.01, dim=0, keepdim=True).to(self.device)
        
    
    def update_starting_states(self, starting_states, global_step, rewarding, total_steps, reward_model_starting_states=None, groundTruth=False):
        if starting_states is not None and len(starting_states) > 0:
            if rewarding:
                starting_states = starting_states.reshape(-1, starting_states.shape[-1])
                if groundTruth:
                    self.ground_truth_starting_states_reward = starting_states
                else:
                    if reward_model_starting_states > self.starting_states_reward_model_max_reward:
                        self.starting_states_reward_model_max_reward = reward_model_starting_states
                        self.starting_states_reward_model = starting_states if len(self.starting_states_reward_model)==0 else np.concatenate((self.starting_states_reward_model,starting_states))
                        self.starting_states_reward_model = self.starting_states_reward_model[-self.cfg.algorithm.safety_args.keepreward_per_model*self.cfg.algorithm.safety_args.parallel_models:]
                self.starting_states_reward = np.concatenate((self.ground_truth_starting_states_reward ,self.starting_states_reward_model))
                if self.starting_states_reward is not None: 
                    plot_starting_states_reward(self.starting_states_reward , global_step, self.cfg.algorithm.env_safety_args, total_steps)
            else:
                self.starting_states_failures = np.concatenate((self.starting_states_failures,starting_states)) if self.starting_states_failures is not None else starting_states
                self.starting_states_failures = self.starting_states_failures[-self.cfg.algorithm.safety_args.keepfailure_per_model*self.cfg.algorithm.safety_args.parallel_models*self.cfg.algorithm.safety_args.keep_eval_iteration_failure_states:]
                plot_starting_states_failure(self.starting_states_failures, global_step, self.cfg.algorithm.env_safety_args,total_steps)
    
    def step_batch(self, state_batch, action_batch):
        action_batch = torch.clamp(action_batch, self.original_low, self.original_high)
        pred_next_obs, pred_rewards, pred_dones, next_model_state = self.model_env.info_step(
            action_batch, {"obs": state_batch}, sample=True
        )
        epistemic_frac = next_model_state["K"]
        conditional_var = next_model_state["conditional_var"]
        lost_info_this_step = torch.clamp(0.5*torch.log2(2*torch.pi*torch.exp(torch.tensor(1))*conditional_var) - self.quantization_bits, torch.tensor(0), torch.inf)
        too_much_single_step_loss = (epistemic_frac > self.per_step_epistemic_frac).sum(dim=1,keepdim=True)>self.cfg.algorithm.safety_args.maximum_number_dimensions_exceeded
        return pred_next_obs, pred_dones,too_much_single_step_loss, lost_info_this_step, pred_rewards, epistemic_frac

    def reset_batch(self, eval_mode = False, batch_size=1):
        start_state = self.starting_state_fn(batch_size)
        random_permuation = np.random.permutation(len(start_state))
        default_starting_states_indices = random_permuation
        if self.cfg.algorithm.get("only_default_starting_states", False)==False:
            number_reward = int(self.cfg.algorithm.safety_args.fraction_reward_states * batch_size)
            number_failure = int(self.cfg.algorithm.safety_args.fraction_failure_states * batch_size)
            number_normal = batch_size - number_failure - number_reward
            assert number_reward + number_failure + number_normal == batch_size
            if self.starting_states_reward is None:
                number_failure = number_failure + number_reward
            if self.starting_states_failures is not None and eval_mode == False:
                indices_failures = np.random.choice(len(self.starting_states_failures), number_failure, replace=True)
                start_state[-number_failure:,:] = self.starting_states_failures[indices_failures]
            if self.starting_states_reward is not None:
                indices_reward = np.random.choice(len(self.starting_states_reward), number_reward, replace=True)
                start_state[number_normal:number_normal+number_reward,:] = self.starting_states_reward[indices_reward]

            
            if eval_mode:
                if self.starting_states_reward is not None:
                    original_default_indices = np.concatenate([np.arange(number_normal), np.arange(number_normal+number_reward, batch_size)])
                    default_starting_states_indices = np.where(np.isin(random_permuation, original_default_indices))[0]
                else:
                    default_starting_states_indices = random_permuation
            else:
                default_starting_states_indices = []
            start_state = start_state[random_permuation]
        start_state_torch =  torch.from_numpy(start_state).to(self.device)
        
        return start_state_torch, {"default_starting_states_indices": default_starting_states_indices}