import random
import time
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
import omegaconf
from dynasaur.util.replay_buffer_nstep import ReplayBufferNStep, ReplayBufferSamples #These are replay buffers with n-step return support
import wandb
from dynasaur.env.gymnasium_hyperplane_safety_wrapper_old_param import HyperplaneSafetyWrapperEnvOldParam as HyperplaneSafetyWrapperEnv
from dynasaur.util.plotting import plot_hyperplanes, plot_eval_trajectories, plot_pink_real
from dynasaur.util.pink_noise import PinkNoiseGenerator, PinkNoiseGeneratorIndividualTrajectories
from dynasaur.util.data_management import save_data_compressed
from dynasaur.models.td3_networks_old_param import TD3Networks
from dynasaur.algorithms.fast_td3_agent_old_param import FastTD3AgentOldParam as FastTD3Agent
import copy

def train(
    env: HyperplaneSafetyWrapperEnv,
    test_env: HyperplaneSafetyWrapperEnv,
    cfg: omegaconf.DictConfig,
    standalone: bool,
    old_networks: TD3Networks = None,
    completeLoad: bool = False,
    global_step_old: int = 0,
    perf_agent = None,
    real_env: gym.Env = None,
    total_safety_env_steps: int = 0,
    termination_fn_real = None,
    index_dynasor: int = 0
) -> np.float32:
    local_eval_index = 0
    one_d_actionspace = cfg.algorithm.env_safety_args.one_d_actionspace
    one_d_extra_visuals = cfg.algorithm.env_safety_args.one_d_extra_visuals
    if cfg.track_wandb and standalone:
        wnb_cfg = OmegaConf.to_container(
            cfg, resolve=True,
        )

        wandb.init(
            project=cfg.wandb_project,
            config=wnb_cfg,
            job_type="train",
            group=cfg.algorithm.name,
            name=cfg.experiment
        )
    if not standalone:
        args = copy.deepcopy(cfg.algorithm.safety_args)
    else:
        args = copy.deepcopy(cfg.algorithm)

    run_name = f"{cfg.experiment}__{cfg.seed}__{int(time.time())}"

    if standalone:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    debug_mode = (cfg.get("debug_mode", False) and cfg.track_wandb)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.algorithm.cuda else "cpu")

    fast_td3_agent = FastTD3Agent(env.observation_space,env.action_space_control, device, cfg, old_networks, hyperplane_exploration=True, completeLoad = completeLoad)

    rb = ReplayBufferNStep(
        args.buffer_size_per_model * args.parallel_models,
        env.observation_space,
        env.action_space_old_param,
        device,
        n_envs=args.parallel_models,
        gamma= args.gamma,
        num_steps=args.num_steps
    )

    rb_eval = ReplayBufferNStep(
        args.max_batch_addition_per_eval_per_model*args.parallel_models,
        env.observation_space,
        env.action_space_old_param,
        device,
        n_envs=1,
        gamma= args.gamma,
        num_steps=args.num_steps
    )

    obs, _ = env.reset_batch()

    action_noise_perf_generator = PinkNoiseGeneratorIndividualTrajectories(batchsize=env.parallel_models,pink_noise_shape=(env.action_space_control.shape[0],), pink_noise_length=cfg.algorithm.safety_args.training_episode_length, device=device, seed=cfg.seed+1)
    action_noise_hyperplanes_generator = PinkNoiseGeneratorIndividualTrajectories(batchsize=env.parallel_models,pink_noise_shape=(env.action_space_control.shape[0]+1,), pink_noise_length=cfg.algorithm.safety_args.training_episode_length, device=device, seed=cfg.seed+2)
    action_noise_hyperplanes_norm_scaling_generator = PinkNoiseGeneratorIndividualTrajectories(batchsize=env.parallel_models,pink_noise_shape=(1,), pink_noise_length=cfg.algorithm.safety_args.training_episode_length, device=device, seed=cfg.seed+3)
    action_noise_eval_generator = PinkNoiseGenerator(pink_noise_shape=(cfg.algorithm.safety_args.eval_episodes,env.action_space_dim), pink_noise_length=cfg.algorithm.trial_length, device=device, seed=cfg.seed+4)
    pink_noise_action_generator_plot_pink_real = PinkNoiseGenerator(pink_noise_shape=(test_env.action_space_dim,),pink_noise_length=test_env.safe_episode_length, device=fast_td3_agent.device, seed=cfg.seed+5)

    max_real_length = 0
    max_eval_length_reached = 0
    max_eval_length_reached_rewarding_states = 0
    counter_safety_patience = 0
    max_eval_rewarding_trajectories_reward_total = -torch.inf
    global_step = global_step_old
    for safety_filter_training_step in range(args.total_timesteps):
        action_start_time = time.time()
        if safety_filter_training_step < args.learning_starts:
                action = env.action_space.sample()
                action = torch.tensor(action, device=device).float()
                if not one_d_actionspace:
                   current_norm =  torch.linalg.norm(action, dim=1, keepdim=True)
                   new_norm = torch.rand(action.shape).to(device)
                   direction = action / current_norm
                   action = direction * new_norm
        else:
            if one_d_actionspace:
                with torch.no_grad():
                    action = fast_td3_agent.act(obs.to(torch.float32))
                    action = torch.clamp(action + action_noise_hyperplanes_norm_scaling_generator.get_pink_action()*env.noise_scales.unsqueeze(1), -1.0, 1.0)
            else:
                with torch.no_grad():
                    action = fast_td3_agent.explore_hyperplane(obs.to(torch.float32), 
                                                    action_noise_hyperplanes_generator.get_pink_action(), 
                                                    env.noise_scales)
        if perf_agent is None:
            next_obs, reward, terminated, truncated,_ = env.step_batch(obs, action) 
        else:
           next_obs, reward, terminated, truncated,_ = env.step_batch(obs, action, action_proposal_perf_batch = perf_agent.explore(obs.to(torch.float32),eps_action=action_noise_perf_generator.get_pink_action(), noise_scales=torch.ones((1,), device=device, dtype=torch.float32)*cfg.algorithm.perf_args.exploration_noise_real_env_perf)) 
        total_safety_env_steps+=1
        rb.add(obs.to(torch.float32).squeeze(), next_obs.to(torch.float32).squeeze(), action.squeeze(), reward.squeeze(), terminated.squeeze(), truncated.squeeze(), {})
        obs = next_obs

        if torch.logical_or(terminated[0], truncated[0]):
            if cfg.track_wandb:
                wandb.log({
                    "Safety_Filter_Learning-Training/AverageReward": env.episode_return[0]/env.episode_length[0],
                    "Safety_Filter_Learning-Training/EpisodicReturn": env.episode_return[0],
                    "Safety_Filter_Learning-Training/EpisodicLength": env.episode_length[0],
                    "Safety_Filter_Learning-Training/total_safety_env_steps": total_safety_env_steps,
                    "Safety_Filter_Learning-Training/global_step": global_step,
                    "Safety_Filter_Learning-Training/safety_filter_training_step": safety_filter_training_step
                }, step=global_step)
            else: 
               print(f"safety_filter_training_step={safety_filter_training_step}, episodic_return={env.episode_return[0]}, episodic_length={env.episode_length[0]}, average_reward={env.episode_return[0]/env.episode_length[0]}")
        
        done_tensor = torch.logical_or(terminated, truncated)
        done_tensor_numpy = done_tensor.cpu().numpy()
        obs,_ = env.reset_batch(states_to_replace_maybe=obs, done_tensor=done_tensor)
        indices_reset_pink = np.arange(done_tensor_numpy.shape[0])[done_tensor_numpy.squeeze()]
        action_noise_hyperplanes_norm_scaling_generator.reset_pink(indices_reset_pink)
        action_noise_hyperplanes_generator.reset_pink(indices_reset_pink)
        action_noise_perf_generator.reset_pink(indices_reset_pink)

        print(f"Env transition time at step {safety_filter_training_step}: {time.time()-action_start_time:.6f} seconds") if (safety_filter_training_step) % cfg.algorithm.log_interval == 0 else None

        if safety_filter_training_step == args.learning_starts-1:
            plot_hyperplanes(global_step, test_env, fast_td3_agent, cfg.algorithm.env_safety_args, total_safety_env_steps) if debug_mode and one_d_actionspace and one_d_extra_visuals else None
            plot_pink_real(global_step, test_env, real_env, fast_td3_agent, cfg.seed, cfg.algorithm.env_safety_args, total_safety_env_steps, pink_noise_action_generator_plot_pink_real, termination_fn_real)
            if perf_agent is not None:
                plot_pink_real(global_step, test_env, real_env, fast_td3_agent, cfg.seed, cfg.algorithm.env_safety_args, total_safety_env_steps, pink_noise_action_generator_plot_pink_real, termination_fn_real, perf_agent, cfg.algorithm.perf_args.exploration_noise_real_env_perf)
        if safety_filter_training_step > args.learning_starts:
            action_start_time = time.time()
            data = rb.sample(args.batch_size_per_model*args.parallel_models)
            
            if rb_eval.size() >= args.batch_size_eval_per_model*args.parallel_models:
                data_eval = rb_eval.sample(args.batch_size_eval_per_model*args.parallel_models) 
                merged_data = ReplayBufferSamples(
                    observations = torch.cat([data.observations, data_eval.observations], dim=0),
                    actions = torch.cat([data.actions, data_eval.actions], dim=0),
                    next_observations = torch.cat([data.next_observations, data_eval.next_observations], dim=0),
                    dones = torch.cat([data.dones, data_eval.dones.squeeze()], dim=0),
                    rewards = torch.cat([data.rewards, data_eval.rewards], dim=0),
                    effective_n_steps = torch.cat([data.effective_n_steps, data_eval.effective_n_steps], dim=0).squeeze()
                )
                data = merged_data
            
            fast_td3_agent.train(data, global_step, total_safety_env_steps)
            global_step+=1
            if safety_filter_training_step % cfg.algorithm.log_interval == 0:  # Print every 100 steps to avoid spam
                print(f"Training time at step {safety_filter_training_step}: {time.time() - action_start_time:.6f} seconds")

            if safety_filter_training_step % (args.eval_after_step*1) == 0:    
                averageEpsidodeLengthPink = plot_pink_real(global_step, test_env, real_env, fast_td3_agent, cfg.seed, cfg.algorithm.env_safety_args,total_safety_env_steps, pink_noise_action_generator_plot_pink_real, termination_fn_real)
                if perf_agent is not None:
                    plot_pink_real(global_step, test_env, real_env, fast_td3_agent, cfg.seed, cfg.algorithm.env_safety_args,total_safety_env_steps, pink_noise_action_generator_plot_pink_real, termination_fn_real, perf_agent, cfg.algorithm.perf_args.exploration_noise_real_env_perf)
                plot_hyperplanes(global_step, test_env, fast_td3_agent, cfg.algorithm.env_safety_args, total_safety_env_steps) if debug_mode and one_d_actionspace and one_d_extra_visuals else None
            
            if safety_filter_training_step % args.eval_after_step == 0:
                if cfg.algorithm.completeSafe:
                    fast_td3_agent.get_td3_networks().save(folder=f"SafetyFilterIntermediate-{index_dynasor}", name=f"SF-BeforeEval-{local_eval_index}.pt")
                    if env.env.starting_states_failures is not None:
                        save_data_compressed(env.env.starting_states_failures, f"SafetyFilterFailureIntermediate-{index_dynasor}/FS-BeforeEval-{local_eval_index}.npz")
                    if env.env.starting_states_reward is not None:
                        save_data_compressed(env.env.starting_states_reward, f"SafetyFilterRewardIntermediate-{index_dynasor}/RS-BeforeEval-{local_eval_index}.npz")
                    local_eval_index+=1
                time_eval_start = time.time()
                obs_eval, info_eval = test_env.reset_batch()
                trajectories_obs = torch.zeros((cfg.algorithm.trial_length+1,cfg.algorithm.safety_args.eval_episodes*2, env.observation_space.shape[0]))
                trajectories_rewards = torch.zeros((cfg.algorithm.trial_length,cfg.algorithm.safety_args.eval_episodes*2))
                trajectories_rewards_control = torch.zeros((cfg.algorithm.trial_length,cfg.algorithm.safety_args.eval_episodes*2))
                trajectories_actions = torch.zeros((cfg.algorithm.trial_length,cfg.algorithm.safety_args.eval_episodes*2, env.action_space_old_param.shape[0]))
                trajectories_actions_qvalue = torch.zeros((cfg.algorithm.trial_length,cfg.algorithm.safety_args.eval_episodes*2))
                trajectories_actions_qvalue_over_threshold_count = torch.zeros((cfg.algorithm.safety_args.eval_episodes*2))
                trajectories_actions_qvalue_over_threshold_return = torch.zeros((cfg.algorithm.safety_args.eval_episodes*2))
                trajectories_actions_qvalue_over_threshold_return_safe_end_index = torch.zeros((cfg.algorithm.safety_args.eval_episodes*2))
                terminated_old = torch.zeros((cfg.algorithm.safety_args.eval_episodes*2,), dtype=torch.bool).to(device)
                truncated_old = torch.zeros((cfg.algorithm.safety_args.eval_episodes*2,), dtype=torch.bool).to(device)
                dict_trajectories_terminated = dict.fromkeys(range(cfg.algorithm.safety_args.eval_episodes * 2), cfg.algorithm.trial_length)
                dict_trajectories_truncated = dict.fromkeys(range(cfg.algorithm.safety_args.eval_episodes * 2), cfg.algorithm.trial_length)
                min_termination_step = cfg.algorithm.trial_length
                min_termination_step_rewarding = cfg.algorithm.trial_length
                trajectories_obs[0] = obs_eval
                action_noise_eval_generator.reset_pink()
                for steps_in_episode_eval in range(cfg.algorithm.trial_length):
                    with torch.no_grad():
                        input_agent_obseval = obs_eval.to(torch.float32)
                        actions = fast_td3_agent.get_target_actor(input_agent_obseval)
                        trajectories_actions_qf1_target = fast_td3_agent.qf1_target(input_agent_obseval, actions)
                        trajectories_actions_qf2_target = fast_td3_agent.qf2_target(input_agent_obseval, actions)
                        trajectories_actions_qvalue[steps_in_episode_eval] = torch.min(trajectories_actions_qf1_target.squeeze(),trajectories_actions_qf2_target.squeeze())
                        trajectories_actions[steps_in_episode_eval] = actions
                    if perf_agent is not None: 
                        action_proposal_perf_batch = perf_agent.explore(obs_eval[:cfg.algorithm.safety_args.eval_episodes].to(torch.float32),eps_action=action_noise_eval_generator.get_pink_action(), noise_scales=torch.ones((1,), device=device, dtype=torch.float32)*cfg.algorithm.perf_args.exploration_noise_real_env_perf) 
                    else:
                        action_proposal_perf_batch = None
                    next_obs, reward, terminated, truncated,reward_control= test_env.step_batch(obs_eval, actions, action_proposal_perf_batch = action_proposal_perf_batch)
                    next_obs = torch.clamp(next_obs, -1000, 1000)
                    trajectories_obs[steps_in_episode_eval+1] = next_obs
                    terminated_new = torch.logical_and(terminated.squeeze(), torch.logical_not(terminated_old.squeeze()))
                    truncated_new = torch.logical_and(truncated.squeeze(), torch.logical_not(truncated_old.squeeze()))
                    terminated_old = torch.logical_or(terminated.squeeze(), terminated_old.squeeze())
                    truncated_old = torch.logical_or(truncated.squeeze(), truncated_old.squeeze())
                    obs_eval = next_obs
                    trajectories_rewards[steps_in_episode_eval] = reward
                    trajectories_rewards_control[steps_in_episode_eval]= reward_control
                    terminated_new_and_not_truncated = torch.logical_and(terminated_new.squeeze(), torch.logical_not(truncated.squeeze()))
                    if truncated_new.any():
                        for index_truncated in torch.where(truncated_new)[0]:
                            if dict_trajectories_truncated[index_truncated.item()] == cfg.algorithm.trial_length:
                                dict_trajectories_truncated[index_truncated.item()] = steps_in_episode_eval
                    if terminated_new_and_not_truncated.any():
                        for index_terminated in torch.where(terminated_new_and_not_truncated)[0]:
                            if dict_trajectories_terminated[index_terminated.item()] == cfg.algorithm.trial_length:
                                dict_trajectories_terminated[index_terminated.item()] = steps_in_episode_eval
                batch_content = 0
                starting_states_failure = []
                default_starting_states_indices = info_eval["default_starting_states_indices"]
                mean_eval_traj_length = np.mean(np.array(list(dict_trajectories_terminated.values()))[default_starting_states_indices])
                if len(default_starting_states_indices)<cfg.algorithm.safety_args.eval_episodes*2:
                    mean_eval_traj_length_rewarding_starting_states = np.mean(np.array(list(dict_trajectories_terminated.values()))[np.setdiff1d(np.arange(len(dict_trajectories_terminated)), default_starting_states_indices)])
                else:
                    mean_eval_traj_length_rewarding_starting_states = -1
                terminated_items = list(dict_trajectories_terminated.items())
                random.shuffle(terminated_items)
                rb_eval = ReplayBufferNStep(
                    args.max_batch_addition_per_eval_per_model*args.parallel_models,
                    env.observation_space,
                    env.action_space_old_param,
                    device,
                    n_envs=1,
                    gamma= args.gamma,
                    num_steps=args.num_steps
                )
                max_reward_trajectory_eval_reward =-torch.inf
                max_reward_trajectory_eval_index = -1
                steps_in_episode_eval_rewarding = -1
                eval_freedom_notion = 0.0
                eval_freedom_notion_rewarding_starting_states=0.0

                for index_terminated, steps_in_episode_eval in terminated_items:
                    steps_in_episode_eval+=1
                    end_of_trajectory = min(dict_trajectories_truncated[index_terminated], steps_in_episode_eval)
                    if index_terminated in default_starting_states_indices:
                        this_trajectory_freedom = torch.sum(torch.linalg.norm(trajectories_actions[0:end_of_trajectory, index_terminated], dim = 1)) / end_of_trajectory
                        eval_freedom_notion+=this_trajectory_freedom
                    else:
                        this_trajectory_freedom = torch.sum(torch.linalg.norm(trajectories_actions[0:end_of_trajectory, index_terminated], dim = 1)) / end_of_trajectory
                        eval_freedom_notion_rewarding_starting_states+=this_trajectory_freedom
                    if index_terminated in default_starting_states_indices: #Meaning this is a truncated trajectory starting from default control starting state
                        #trajectoy was truncated
                        end_of_reward_trajectory = max(min(dict_trajectories_truncated[index_terminated],dict_trajectories_terminated[index_terminated]),1)
                        traj_end_safe = torch.where(trajectories_actions_qvalue[:end_of_reward_trajectory,index_terminated]>cfg.algorithm.safety_args.safetyQThreshold)
                        traj_end_safe = max(traj_end_safe[0]) if len(traj_end_safe[0])>0 else 1
                        trajectories_actions_qvalue_over_threshold_count[index_terminated] = torch.sum(trajectories_actions_qvalue[:end_of_reward_trajectory,index_terminated]>cfg.algorithm.safety_args.safetyQThreshold)
                        end_of_reward_trajectory = min(end_of_reward_trajectory, traj_end_safe) #Conservative End regarding Q function
                        trajectories_actions_qvalue_over_threshold_return[index_terminated] = torch.sum(trajectories_rewards_control[0:end_of_reward_trajectory, index_terminated])
                        trajectories_actions_qvalue_over_threshold_return_safe_end_index[index_terminated] = end_of_reward_trajectory
                    if dict_trajectories_terminated[index_terminated] != cfg.algorithm.trial_length:
                        if batch_content > args.max_batch_addition_per_eval_per_model*args.parallel_models:
                            continue
                        min_termination_step = min(min_termination_step, steps_in_episode_eval) if index_terminated in default_starting_states_indices else min_termination_step
                        min_termination_step_rewarding = min(min_termination_step_rewarding, steps_in_episode_eval) if index_terminated not in default_starting_states_indices else min_termination_step_rewarding
                        start_tracking_fail = 0
                        batch_len = steps_in_episode_eval - start_tracking_fail
                        batch_content+= batch_len
                        obs_bad = trajectories_obs[start_tracking_fail:steps_in_episode_eval, index_terminated]
                        next_obs_bad = trajectories_obs[start_tracking_fail+1:steps_in_episode_eval+1, index_terminated]
                        starting_state_indices = np.arange(max((steps_in_episode_eval-args.maximum_time_to_failure_early), 0),max((steps_in_episode_eval-args.maximum_time_to_failure_late), -1), -(int((args.maximum_time_to_failure_late-args.maximum_time_to_failure_early)/args.keepfailure_per_model)))
                        for starting_state_index in starting_state_indices:
                            starting_states_failure.append(trajectories_obs[int(starting_state_index), index_terminated].detach().cpu().numpy())
                        terminated_bad = torch.zeros((batch_len), dtype=torch.bool)
                        terminated_bad[-1] = True
                        action_bad = trajectories_actions[start_tracking_fail:steps_in_episode_eval, index_terminated]
                        reward_bad = trajectories_rewards[start_tracking_fail:steps_in_episode_eval, index_terminated]
                        reward_bad[-1] = env.termination_penalty
                        rb_eval.add_batch(obs_bad.to(torch.float32), next_obs_bad.to(torch.float32), action_bad, reward_bad, terminated_bad, {})

                if torch.any(trajectories_actions_qvalue_over_threshold_count>50):
                    # Get indices where count > 50
                    valid_indices = torch.where(trajectories_actions_qvalue_over_threshold_count > 50)[0]
                                        
                    # Get the top 10% threshold for count values among valid trajectories
                    valid_counts = trajectories_actions_qvalue_over_threshold_count[valid_indices]
                    top_10_percent_threshold = torch.quantile(valid_counts, 0.9)
                    
                    # Filter to only top 10% trajectories
                    top_10_percent_mask = trajectories_actions_qvalue_over_threshold_count >= top_10_percent_threshold
                    top_10_percent_indices = torch.where(top_10_percent_mask)[0]
                    
                    # Find the trajectory with maximum return among top 10%
                    top_10_percent_returns = trajectories_actions_qvalue_over_threshold_return[top_10_percent_indices]
                    max_return_idx_in_top10 = torch.argmax(top_10_percent_returns, dim=0)

                    max_reward_trajectory_eval_reward = trajectories_actions_qvalue_over_threshold_return[top_10_percent_indices][max_return_idx_in_top10].item()
                    steps_in_episode_eval_rewarding = int(trajectories_actions_qvalue_over_threshold_return_safe_end_index[top_10_percent_indices][max_return_idx_in_top10].item())

                eval_freedom_notion = eval_freedom_notion / len(default_starting_states_indices)
                if len(default_starting_states_indices)<cfg.algorithm.safety_args.eval_episodes*2:
                    eval_freedom_notion_rewarding_starting_states = eval_freedom_notion_rewarding_starting_states / (cfg.algorithm.safety_args.eval_episodes*2 - len(default_starting_states_indices))
                else:
                    eval_freedom_notion_rewarding_starting_states = -1
                if steps_in_episode_eval_rewarding!=-1:
                    eval_rewarding_starting_states = trajectories_obs[:,top_10_percent_indices][:,max_return_idx_in_top10][0:steps_in_episode_eval_rewarding,:].detach().cpu().numpy()
                time_eval_start_2 = time.time()
                starting_states_failure = np.array(starting_states_failure)
                plot_eval_trajectories(global_step, trajectories_actions_qvalue.detach().cpu().numpy(),trajectories_obs.detach().cpu().numpy(),dict_trajectories_terminated, dict_trajectories_truncated, cfg.algorithm.env_safety_args, cfg.algorithm.trial_length, total_safety_env_steps, perf_agent is None,  eval_episodes=cfg.algorithm.safety_args.eval_episodes, safetyQThreshold=cfg.algorithm.safety_args.safetyQThreshold) if debug_mode else None
                print(f"Evaluation time at step {safety_filter_training_step}: {time_eval_start_2 - time_eval_start:.6f} seconds, with plotting time: {time.time() - time_eval_start_2:.6f} seconds")
                env.env.update_starting_states(starting_states_failure, global_step=global_step, rewarding = False, total_steps=total_safety_env_steps)
                total_safety_env_steps+=1
                global_step+=1
                if max_eval_rewarding_trajectories_reward_total< max_reward_trajectory_eval_reward:
                    max_eval_rewarding_trajectories_reward_total = max_reward_trajectory_eval_reward
                    if cfg.track_wandb:
                        wandb.log({
                            "Safety_Filter_Learning-Evaluation/MaxReturnRewardingStates": max_eval_rewarding_trajectories_reward_total,
                            "Safety_Filter_Learning-Evaluation/MaxReturnRewardingStatesLength": steps_in_episode_eval_rewarding,
                            "Safety_Filter_Learning-Evaluation/MeanReturnRewardingStates": max_eval_rewarding_trajectories_reward_total/steps_in_episode_eval_rewarding,
                            "Safety_Filter_Learning-Evaluation/safety_filter_training_step": safety_filter_training_step
                        }, step=global_step)
                    global_step+=1
                    env.env.update_starting_states(eval_rewarding_starting_states, global_step=global_step, rewarding = True, total_steps=total_safety_env_steps, reward_model_starting_states=max_reward_trajectory_eval_reward)
                episode_min_length = min_termination_step
                counter_safety_patience+=1
                print("Episode Batch Min Length", episode_min_length , "global step:", global_step)
                print("Eval Mean Trajectory Length:", mean_eval_traj_length, "global step:", global_step)
                print("Added to Eval Replay Buffer:", batch_content, "transitions.")
                if mean_eval_traj_length>=max_eval_length_reached+ args.patience_threshold_safety :
                    max_eval_length_reached = mean_eval_traj_length
                    counter_safety_patience = 0
                if mean_eval_traj_length_rewarding_starting_states>=max_eval_length_reached_rewarding_states+ args.patience_threshold_safety :
                    max_eval_length_reached_rewarding_states = mean_eval_traj_length_rewarding_starting_states
                    counter_safety_patience = 0
                if cfg.track_wandb:
                    wandb.log({
                        "Safety_Filter_Learning-Evaluation/MinimumLength": episode_min_length,
                        "Safety_Filter_Learning-Evaluation/MinimumLengthRewardingStartingStates": min_termination_step_rewarding,
                        "Safety_Filter_Learning-Evaluation/MeanLength": mean_eval_traj_length,
                        "Safety_Filter_Learning-Evaluation/MeanLengthRewardingStartingStates": mean_eval_traj_length_rewarding_starting_states,
                        "Safety_Filter_Learning-Evaluation/total_safety_env_steps": total_safety_env_steps,
                        "Safety_Filter_Learning-Evaluation/counter_safety_patience":counter_safety_patience,
                        "Safety_Filter_Learning-Evaluation/patience_safety":args.patience_safety,
                        "Safety_Filter_Learning-Evaluation/EvalFreedomNotion": eval_freedom_notion,
                        "Safety_Filter_Learning-Evaluation/EvalFreedomNotionRewardingStartingStates": eval_freedom_notion_rewarding_starting_states,
                        "Safety_Filter_Learning-Evaluation/max_eval_length_reached":max_eval_length_reached,    
                        "Safety_Filter_Learning-Evaluation/safety_filter_training_step": safety_filter_training_step
                            }, step=global_step)
                terminated_in_model = (mean_eval_traj_length < cfg.algorithm.trial_length) or (mean_eval_traj_length_rewarding_starting_states < cfg.algorithm.trial_length)
                if not standalone and (not terminated_in_model or  counter_safety_patience>=args.patience_safety):
                    print("Safety Filter is learned! Start Real Perf Training!")
                    plot_hyperplanes(global_step, test_env, fast_td3_agent,cfg.algorithm.env_safety_args, total_safety_env_steps) if debug_mode and one_d_actionspace and one_d_extra_visuals else None
                    global_step+=1
                    total_safety_env_steps+=1
                    plot_eval_trajectories(global_step, trajectories_actions_qvalue.detach().cpu().numpy(), trajectories_obs.detach().cpu().numpy(),dict_trajectories_terminated, dict_trajectories_truncated, cfg.algorithm.env_safety_args, cfg.algorithm.trial_length, total_safety_env_steps, perf_agent is None, prefix="Final", eval_episodes=cfg.algorithm.safety_args.eval_episodes, safetyQThreshold=cfg.algorithm.safety_args.safetyQThreshold) if debug_mode else None
                    safety_filter_networks = fast_td3_agent.get_td3_networks()
                    return safety_filter_networks, global_step, total_safety_env_steps
    print("Safety Filter has not been learned! Start Model Perf Training!")
    safety_filter_networks = fast_td3_agent.get_td3_networks()
    return safety_filter_networks, global_step, total_safety_env_steps

