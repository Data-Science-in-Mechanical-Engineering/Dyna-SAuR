import imageio
import os
from typing import Optional
import gymnasium as gym
import numpy as np
import omegaconf
import torch
from dynasaur.algorithms.fast_td3_agent import FastTD3Agent
import dynasaur.models as models
import dynasaur.types_dynasaur as types_dynasaur
import dynasaur.util as util
from dynasaur.util.gather_data_walker import gather_data_walker
from dynasaur.util.gather_data_walker import gather_data_walker
from dynasaur.util.prior_controller import LQR_Cartpole
from omegaconf import OmegaConf
import numpy as np
import torch
import os
from dynasaur.algorithms.td3_safety_filter import train as td3_safety_filter
from dynasaur.util.data_management import save_data_compressed, load_data_compressed
from dynasaur.models.wrapper.dynasaur_model_wrapper import dynasaurModelWrapper
from dynasaur.models.wrapper.filtered_agents import FilteredAgent
from dynasaur.util.plotting import plot_starting_states_ground_truth, define_metrics
from dynasaur.env.gymnasium_hyperplane_safety_wrapper import HyperplaneSafetyWrapperEnv
from dynasaur.util.pink_noise import PinkNoiseGenerator
from dynasaur.util.replay_buffer_nstep import ReplayBufferNStep
import dynasaur.util.common

def rollout_model_and_populate_perf_model_buffer(
                model_env,
                real_perf_replay_buffer,
                filtered_agent,
                model_perf_replay_buffer,
                rollout_length_perf_model_rollouts,
                buffer_size_per_model,
                parallel_models,
                track_wandb,
                global_step,
                cfg,
                wandb,
                total_perf_env_steps,
                pink_noise_exploration_generator):
    # Populate Dπ,p̂ with model-based rollouts for control policy learning (Algorithm 6, Appendix A.3).
    # Rollouts are truncated when leaving the certain set E (eq. 25): per-step entropy exceeds λ1
    # or accumulated information loss exceeds λ2, keeping rollouts within the accurate model region.
    with torch.no_grad():
        num_added = 0
        batch_size = buffer_size_per_model*parallel_models
        obs_dataset = real_perf_replay_buffer.observations[:real_perf_replay_buffer.pos][:,0,:]
        actions_dataset = real_perf_replay_buffer.actions[:real_perf_replay_buffer.pos][:,0,:]
        model_state = {"obs": obs_dataset}
        pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
            actions_dataset, model_state, sample=True
        )
        quantization_bits = torch.log2(torch.tensor(1e-6))
        lost_info_dataset = torch.clamp(0.5*torch.log2(2*torch.pi*torch.exp(torch.tensor(1))*next_model_state["conditional_var"]) - quantization_bits, 0, torch.inf)
        # λ1 threshold (per-step epistemic fraction, ζ1=0.99 quantile) and
        # λ2 threshold (accumulated info loss, ζ2=0.01 quantile × horizon) define the certain set E (eq. 23, 25).
        per_step_epistemic_frac_threshold = torch.quantile(next_model_state["K"], q=0.99, dim=0, keepdim=True)
        max_infoloss_threshold = rollout_length_perf_model_rollouts*torch.quantile(lost_info_dataset, q=0.01, dim=0, keepdim=True)

        total_initial_states = 0
        # repeat till we have added enough transitions
        rollout_tracker = []
        rollout_tracker_idx = -1
        pink_noise_exploration_generator.reset_pink()
        while num_added < batch_size:
            total_initial_states += parallel_models
            batch = real_perf_replay_buffer.sample(parallel_models) 
            obs = batch.observations
            obs_dims = obs.shape[-1]
            lost_info_acummulated = torch.zeros((parallel_models, obs_dims), device=cfg.device) # lost info for each obs dim and reward
            curr_idx = torch.arange(parallel_models).to(cfg.device)
            explore_noise_scales = torch.rand((cfg.algorithm.perf_args.parallel_models,), device=cfg.device)*(cfg.algorithm.perf_args.max_noise_scale - cfg.algorithm.perf_args.min_noise_scale) + cfg.algorithm.perf_args.min_noise_scale
            rollout_tracker_idx+=1
            rollout_tracker.append(torch.zeros(parallel_models, device=cfg.device))
            for step in range(rollout_length_perf_model_rollouts):
                accum_dones = torch.zeros(obs.shape[0],dtype=bool, device=cfg.device)
                action,perf_action, penalty = filtered_agent.act_eps_penalty(obs, pink_noise_exploration_generator.get_pink_action()[curr_idx],explore_noise_scales[curr_idx], random_explore=False)
                action = action.detach().cpu().numpy()
                if len(action.shape)==1:
                    action=action[None,:]
                if len(perf_action.shape)==1:
                    perf_action=perf_action.unsqueeze(0)
                pred_next_obs, pred_rewards, pred_dones, next_model_state = model_env.info_step(
                    action, {"obs": obs}, sample=True
                )
                # Penalize control agent for deviating from the safety-filtered action: r_t - ||a_t - a^V_t||_2 (Appendix A.3).
                pred_rewards-=penalty.squeeze()*cfg.algorithm.penaly_factor_safety_filter_violation
                truncateds = torch.zeros_like(pred_dones, dtype=bool)
                lost_info_this_step = torch.clip(0.5*torch.log2(2*torch.pi*torch.exp(torch.tensor(1))*next_model_state["conditional_var"]) - quantization_bits, 0, torch.inf)
                lost_info_acummulated += lost_info_this_step
                epistemic_frac = next_model_state["K"]
                # Terminate rollout if the model leaves the certain set E: λ1 (per-step) or λ2 (accumulated) violated (eq. 25).
                too_much_single_step_loss = (epistemic_frac > per_step_epistemic_frac_threshold).sum(dim=1,keepdim=True)>cfg.algorithm.perf_args.maximum_number_dimensions_exceeded
                accum_dones |= too_much_single_step_loss.squeeze()
                accum_dones |= (lost_info_acummulated > max_infoloss_threshold).any(axis=-1)
                if (~accum_dones).sum()==0:
                    break
                # Store unfiltered action a_t in Dπ,p̂ (not filtered a^V_t) so the Q-function
                # learns to credit the control policy's own action choice (Algorithm 6).
                model_perf_replay_buffer.add_batch(
                    obs[~accum_dones],
                    pred_next_obs[~accum_dones],
                    perf_action[~accum_dones],
                    pred_rewards[~accum_dones],
                    pred_dones[~accum_dones, 0],
                    truncateds[~accum_dones, 0],
                )
                rollout_tracker[rollout_tracker_idx][(curr_idx)]+=1
                num_added += ((~accum_dones).sum())
                if num_added > batch_size:
                    break
                accum_dones |= pred_dones.squeeze()
                obs = pred_next_obs[~accum_dones] 
                if len(obs) == 0: break
                lost_info_acummulated = lost_info_acummulated[~accum_dones]
                curr_idx = curr_idx[~accum_dones]
        if track_wandb:
            # wandb logging for rollout metrics
            rollout_tracker = torch.cat([rollout_tracker_element for rollout_tracker_element in rollout_tracker])
            data = {
                "rollout/global_step": global_step,
                "rollout/average_length": rollout_tracker.mean(),
                "rollout/minimum_length": rollout_tracker.min(),
                "rollout/maximum_length": rollout_tracker.max(),
                "rollout/added_transitions": num_added,
                "rollout/total_number_rollouts": total_initial_states,
                "rollout/total_perf_env_steps": total_perf_env_steps
            }
            for obs_dim in range(obs_dims):
                data["rollout/loss_rate_limit_" + str(obs_dim)] = per_step_epistemic_frac_threshold[0, obs_dim]
                data["rollout/loss_limit_" + str(obs_dim)] = max_infoloss_threshold[0, obs_dim]
            data["rollout/quant_ent"] = quantization_bits
            hist_data = rollout_tracker.detach().cpu().numpy()
            hist = np.histogram(hist_data, bins=min(512, int(rollout_tracker.max())), density=True,)
            data["rollout/rollout_length_histogram"] = wandb.Histogram(np_histogram=hist)
            wandb.log(
                data=data, step=global_step
            )
        return num_added
    


def train(
    cfg: omegaconf.DictConfig,
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: types_dynasaur.TermFnType,
    reward_fn:types_dynasaur.RewardFnType,
    starting_state_fn,
    termination_fn_model: types_dynasaur.TermFnType,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    
    # ------------------- Initialization -------------------
    global_step = 0
    total_perf_env_steps = 0
    one_d_actionspace = cfg.algorithm.env_safety_args.one_d_actionspace
    one_d_extra_visuals = cfg.algorithm.env_safety_args.one_d_extra_visuals
    wandb = None
    debug_mode = (cfg.get("debug_mode", False) and cfg.track_wandb)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    torch_generator.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.algorithm.cuda else "cpu")

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    if cfg.track_wandb:
        import wandb

        wnb_cfg = OmegaConf.to_container(
            cfg, resolve=True,
        )

        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.wandb_project,

            # track hyperparameters and run metadata
            config=wnb_cfg,
            job_type="train",
            group=cfg.algorithm.name,
            name=cfg.experiment
        )

        define_metrics(obs_shape)
        
    fast_td3_agent_perf = FastTD3Agent(env.observation_space,env.action_space, device, cfg, None, hyperplane_exploration=False)
    fast_td3_agent_perf.get_td3_networks().save(folder="PerformanceAgentCompleted", name=f"PA-{0}.pt")
    work_dir = work_dir or os.getcwd()
    logger = util.Logger(work_dir)

    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32

    replay_buffer = util.common.create_replay_buffer_fixed_capacity(
        cfg,
        cfg.algorithm.prior_replaybuffer_size + cfg.algorithm.max_double_learning_iters*cfg.algorithm.max_number_perf_steps_per_mesorrl_iteration,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    eval_replay_buffer = util.common.create_replay_buffer_fixed_capacity(
        cfg,
        cfg.algorithm.max_double_learning_iters*(cfg.algorithm.trial_length*cfg.algorithm.perf_real_env_evaluation_episodes),
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    if cfg.algorithm.get("load_replay_buffer", False) == False:
        # Collect prior dataset D₀^{p,p̂} using a stabilizing controller perturbed with pink noise
        # (Appendix D.1). For CartPole an LQR is used; for Walker a partially trained RL agent.
        # This initial data is required to train the first dynamics model and derive the initial certain set E₀.
        if cfg.algorithm.name=="dynasaur_cartpole":
            agent_lqr = LQR_Cartpole(cfg)
            util.common.gatherInitialData(
                env,
                cfg.algorithm.prior_replaybuffer_size,
                agent_lqr,
                replay_buffer,
                cfg
            )
        elif cfg.algorithm.name=="dynasaur_walker":
            gather_data_walker(env, cfg, replay_buffer)
        (
            obs_loaded,
            action_loaded,
            next_obs_loaded,
            reward_loaded,
            terminated_loaded,
            truncated_loaded,
        ) = replay_buffer.get_all().astuple()
        save_data_compressed(obs_loaded, "obs.npz")
        save_data_compressed(action_loaded, "action.npz")
        save_data_compressed(next_obs_loaded, "next_obs.npz")
        save_data_compressed(reward_loaded, "reward.npz")
        save_data_compressed(terminated_loaded, "terminated.npz")
        save_data_compressed(truncated_loaded, "truncated.npz")
        save_data_compressed(replay_buffer, "replay_buffer_initial.pkl")
    else:
        replay_buffer_loaded = load_data_compressed(cfg.algorithm.load_replay_buffer_path)
        (
            obs_loaded,
            action_loaded,
            next_obs_loaded,
            reward_loaded,
            terminated_loaded,
            truncated_loaded,
        ) = replay_buffer_loaded.get_all().astuple()
        

        print("Successfully saved Replay Buffer components as npz files.")

        replay_buffer.add_batch(
            obs_loaded,
            action_loaded,
            next_obs_loaded,
            reward_loaded,
            terminated_loaded,
            truncated_loaded,
        )
        cfg.algorithm.prior_replaybuffer_size = replay_buffer_loaded.num_stored
        print(f"Loaded replay buffer with {replay_buffer_loaded.num_stored} samples")
        del replay_buffer_loaded

    replay_buffer_obs_all = replay_buffer.obs[:replay_buffer.num_stored,:]
    plot_starting_states_ground_truth(replay_buffer_obs_all,replay_buffer_obs_all[:2],replay_buffer_obs_all[:2], global_step, cfg.algorithm.env_safety_args, total_perf_env_steps) if (debug_mode) else None
    del replay_buffer_obs_all
    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    dynamics_model = util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    if cfg.algorithm.get("load_dynamics", False):
        dynamics_model = load_data_compressed(cfg.algorithm.load_dynamics_path)
        print("Loaded dynamics model!")
    model_env = models.ModelEnv(
        env, dynamics_model, termination_fn_model, reward_fn, generator=torch_generator
    )
    model_trainer = models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.algorithm.model_lr,
        weight_decay=cfg.algorithm.model_wd,
        logger=None if silent else logger,
        track_wandb = cfg.track_wandb,
        wandb = wandb
    )
    if cfg.algorithm.get("load_dynamics", False) == False:
        # Train initial Probabilistic Ensemble model p̂₀ on D₀^{p,p̂} by minimizing negative log-likelihood
        # (Algorithm 3). This establishes the initial certain set E₀ used for safety filter training.
        print("Starting model training...")
        global_step = util.common.train_model_and_save_model_and_data(
                                                                global_step,
                                                                dynamics_model,
                                                                model_trainer,
                                                                cfg.algorithm,
                                                                replay_buffer,
                                                                work_dir=None,
                                                            )

        save_data_compressed(dynamics_model, "ModelCompleted/M-0.pkl")
    total_safety_env_steps = 0
    dynasaur_model_wrapper = dynasaurModelWrapper(model_env=model_env,replay_buffer=replay_buffer, cfg=cfg, env=env, device=device, starting_state_fn = starting_state_fn)
    # Wrap model with hyperplane safety filter MDP M^SF (Section 5.1): the filter policy µ outputs
    # hyperplane actions u_t ∈ U (unit hyperball, eq. 13) that are mapped to (w_t, b_t) via bijection h (eq. 14).
    dynasaur_model = HyperplaneSafetyWrapperEnv(dynasaur_model_wrapper, cfg, eval_mode = False)
    dynasaur_model_test = HyperplaneSafetyWrapperEnv(dynasaur_model_wrapper, cfg, eval_mode = True)
    total_max_reward_starting=-np.inf
    total_max_reward_starting_states = None

    if one_d_actionspace and one_d_extra_visuals:
        dynasaur_model.env.visualize_accurate_set(index_dynasaur=-1, global_step=global_step, total_perf_env_steps=total_perf_env_steps)

    # Learn initial filter policy µ₀ using model-based rollouts within E₀ (Algorithm 4/5).
    # Filter training loops until evaluation confirms the filter keeps the behavior policy safe.
    safety_networks, global_step, total_safety_env_steps = td3_safety_filter(dynasaur_model, dynasaur_model_test, cfg, standalone=False, old_networks = None ,global_step_old = global_step, perf_agent=None, real_env=env, total_safety_env_steps=total_safety_env_steps,  termination_fn_real=termination_fn, index_dynasor=0)
    safety_networks.save(folder="SafetyFilterCompleted", name="SF-0.pt")
    print("Safety Filter stored!")

    safety_filte_fast_td3_agent = FastTD3Agent(dynasaur_model.observation_space,dynasaur_model.action_space_control, device, cfg, safety_networks, hyperplane_exploration=True, completeLoad=True)
    print("Filtered Agent Created")
    
    # Outer Dyna-SAuR loop (Algorithm 2): each iteration j retrains both policies from scratch
    # to avoid primacy bias, then expands E by collecting new environment data under µ.
    DoubleLearningIters=cfg.algorithm.max_double_learning_iters
    countTerminated=0
    countTerminated_eval = 0
    perf_episodes = 0


    pink_noise_exploration_generator = PinkNoiseGenerator(pink_noise_shape=(dynasaur_model.action_space_dim,), pink_noise_length=cfg.algorithm.trial_length, device=device,seed = cfg.seed) 
    pink_noise_exploration_generator_model_rollout = PinkNoiseGenerator(pink_noise_shape=(cfg.algorithm.perf_args.parallel_models,dynasaur_model.action_space_dim), pink_noise_length=cfg.algorithm.rollout_length_perf_model_rollouts, device=model_env.device, seed = cfg.seed+1) 
    initial_exploration_steps_perf = cfg.algorithm.initial_exploration_steps_perf
    max_number_perf_steps = cfg.algorithm.max_number_perf_steps_per_mesorrl_iteration
    perf_updates_every_steps = cfg.algorithm.perf_updates_every_steps
    maxReturnEval = -np.inf
    for k in range(DoubleLearningIters):
        num_stored_old = replay_buffer.num_stored
        num_stored_old_eval = eval_replay_buffer.num_stored
        print(f"=================== MESORL Iteration {k+1} ===================")
        Return = 0
        ReturnControl = 0
        agents_old_perf_networks = fast_td3_agent_perf.get_td3_networks()

        fast_td3_agent_perf = FastTD3Agent(env.observation_space,env.action_space, device, cfg, agents_old_perf_networks, hyperplane_exploration=False, completeLoad=cfg.algorithm.completeLoading)

        # κ^π_µ: compose π with µ — actions proposed by π are projected to the closest viable action
        # satisfying w_t^T a ≥ b_t (eq. 7).
        filtered_agent = FilteredAgent(fast_td3_agent_perf,safety_filte_fast_td3_agent, device, dynasaur_model)
        video_frames=[]
        terminated = False
        truncated = False
        number_perf_env_steps = 0
        obs, _ = env.reset()
        obs = torch.tensor(obs,device=device,dtype=torch.float32)
        # Dπ,p̂: model control buffer, populated by Infoprop rollouts under the filtered policy.
        model_perf_replay_buffer = ReplayBufferNStep(
            cfg.algorithm.perf_args.batch_size_per_model * cfg.algorithm.perf_args.parallel_models * cfg.algorithm.number_model_rollouts_keep_data ,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            gamma= cfg.algorithm.perf_args.gamma,
            num_steps=1
        )

        # Dπ,p: environment control buffer, populated by real filtered environment interactions.
        real_perf_replay_buffer = ReplayBufferNStep(
            max_number_perf_steps+1,
            env.observation_space,
            env.action_space,
            device,
            n_envs=1,
            gamma= cfg.algorithm.perf_args.gamma,
            num_steps=1
        )

        countTerminated_old = countTerminated
        steps_of_episode = 0
        while  number_perf_env_steps < max_number_perf_steps and countTerminated-countTerminated_old<cfg.algorithm.max_terminated_episodes_per_mesorrl_iteration:
            number_perf_env_steps+=1
            random_explore = number_perf_env_steps<initial_exploration_steps_perf
            action,perf_action, penalty = filtered_agent.act_eps_penalty(obs, pink_noise_exploration_generator.get_pink_action(), noise_scale=torch.ones((1,), device=device, dtype=torch.float32)*cfg.algorithm.perf_args.exploration_noise_real_env_perf, random_explore=random_explore)
            action = action.detach().cpu().numpy()
            next_obs, _, _, truncated, info = env.step(action)
            terminated = termination_fn(torch.tensor(action,device=device,dtype=torch.float32).unsqueeze(0), torch.tensor(next_obs,device=device,dtype=torch.float32).unsqueeze(0)).squeeze().detach().cpu().numpy()
            control_reward = reward_fn(torch.tensor(action,device=device,dtype=torch.float32).unsqueeze(0), torch.tensor(next_obs,device=device,dtype=torch.float32).unsqueeze(0)).squeeze()
            steps_of_episode+=1

            total_perf_env_steps+=1
            penalized_reward=control_reward - (penalty*cfg.algorithm.penaly_factor_safety_filter_violation)

            if number_perf_env_steps==initial_exploration_steps_perf:
                truncated = True  # Force end of episode to reset exploration noise
            
            # Store filtered action a^V_t in Dp,p̂ (model learning buffer) since that is what was executed.
            # Store unfiltered action a_t with penalty in Dπ,p (control learning buffer) so π learns
            # to propose actions close to the viable set, reducing filter intervention over time.
            replay_buffer.add(obs.detach().cpu().numpy(), action, next_obs, control_reward.detach().cpu().numpy(), terminated, truncated)
            next_obs = torch.tensor(next_obs,device=device,dtype=torch.float32)
            real_perf_replay_buffer.add(obs,next_obs ,perf_action, penalized_reward, torch.tensor(terminated,device=device,dtype=torch.float32), torch.tensor(truncated,device=device,dtype=torch.float32), {})

            frame = env.render()  # Get frame
            video_frames.append(frame)
            Return+=penalized_reward
            ReturnControl+=control_reward
            obs = next_obs
            if terminated or truncated:
                if terminated:
                    print("Terminated at Step of Episode:", steps_of_episode)
                    countTerminated+=1
                imageio.mimsave("cartpole.mp4", video_frames, fps=30)  # Adjust FPS as needed
                # Log video to wandb
                global_step += 1
                perf_episodes +=1
                video_frames=[]
                
                wandb.log({
                            "PerformanceAgentTraining-Train/PerfEpisodes": perf_episodes,
                        "PerformanceAgentTraining-Train/CountTerminated":countTerminated,
                            "PerformanceAgentTraining-Train/AverageReturnDuringPerfTraining":Return,
                            "PerformanceAgentTraining-Train/AverageControlReturnDuringPerfTraining":ReturnControl,
                            "PerformanceAgentTraining-Train/TotalPerfEnvSteps":total_perf_env_steps,
                            "PerformanceAgentTraining-Train/Training-Video": wandb.Video("cartpole.mp4", fps=30, format="mp4")
                        }, step = global_step)
                # Track τ_G: states from high-return non-failed episodes are used to shape ρ^SF_0
                # (eq. 20, ν3 component), ensuring the filter stays permissive in rewarding regions.
                if (ReturnControl)>total_max_reward_starting and not terminated:
                    total_max_reward_starting=ReturnControl
                    total_max_reward_starting_states = real_perf_replay_buffer.observations[real_perf_replay_buffer.pos - steps_of_episode : real_perf_replay_buffer.pos,:]
                    total_max_reward_starting_states = total_max_reward_starting_states[:,0,:].detach().cpu().numpy()
                Return = 0
                ReturnControl = 0
                obs, _ = env.reset()
                obs = torch.tensor(obs,device=device,dtype=torch.float32)
                terminated = False
                truncated = False   
                steps_of_episode = 0
                pink_noise_exploration_generator.reset_pink()

            if (number_perf_env_steps + 1) % cfg.algorithm.freq_train_model == 0 and (number_perf_env_steps+1>=initial_exploration_steps_perf):
                global_step += 1
                pink_noise_exploration_generator_model_rollout.reset_pink()
                num_added = rollout_model_and_populate_perf_model_buffer(
                    model_env,
                    real_perf_replay_buffer,
                    filtered_agent,
                    model_perf_replay_buffer,
                    cfg.algorithm.rollout_length_perf_model_rollouts,
                    cfg.algorithm.perf_args.batch_size_per_model,
                    cfg.algorithm.perf_args.parallel_models,
                    cfg.track_wandb,
                    global_step,
                    cfg,
                    wandb if cfg.track_wandb else None,
                    total_perf_env_steps,
                    pink_noise_exploration_generator_model_rollout
                )
                global_step += 1
                print(f"Added {num_added} to Model Replay Buffer")

            if number_perf_env_steps%perf_updates_every_steps==0 and (number_perf_env_steps+1>=initial_exploration_steps_perf):
                for _ in range(cfg.algorithm.perf_updates_per_step):
                    # Mix real environment data and synthetic model data (Dyna-style, Algorithm 6).
                    use_real_data = (rng.random() < cfg.algorithm.real_data_ratio)
                    if  not use_real_data and (model_perf_replay_buffer.n_envs* model_perf_replay_buffer.pos < cfg.algorithm.perf_args.batch_size_per_model*cfg.algorithm.perf_args.parallel_models and not model_perf_replay_buffer.full):
                        print("Not enough model data to update")
                        continue
                    if  use_real_data and real_perf_replay_buffer.n_envs* real_perf_replay_buffer.pos < cfg.algorithm.perf_args.batch_size_per_model:
                        print("Not enough real data to update")
                        continue  
                    which_buffer = real_perf_replay_buffer if use_real_data else model_perf_replay_buffer
                    batch_size_update = cfg.algorithm.perf_args.batch_size_per_model if use_real_data else cfg.algorithm.perf_args.batch_size_per_model#*cfg.algorithm.perf_args.parallel_models
                    data = which_buffer.sample(batch_size_update)
                    fast_td3_agent_perf.train(data, global_step, total_perf_env_steps)
                    global_step+=1
        if number_perf_env_steps == max_number_perf_steps:
            print(f"Reached max number of Perf updates {max_number_perf_steps}")
        obs_eval, _ = test_env.reset()
        perf_real_env_evaluation_episodes = cfg.algorithm.perf_real_env_evaluation_episodes
        eval_episode_counter= 0
        ReturnEval = 0
        terminated_eval = False
        truncated_eval = False
        video_frames_eval = []
        while eval_episode_counter < perf_real_env_evaluation_episodes:
            if terminated_eval or truncated_eval:
                obs_eval, _ = test_env.reset()
                terminated_eval = False
                truncated_eval = False 
                eval_episode_counter+=1  

            action_eval = filtered_agent.act(obs_eval)
            next_obs_eval, _, _, truncated_eval, info_eval = test_env.step(action_eval)
            terminated_eval = termination_fn(torch.tensor(action_eval,device=device,dtype=torch.float32).unsqueeze(0), torch.tensor(next_obs_eval,device=device,dtype=torch.float32).unsqueeze(0)).squeeze().detach().cpu().numpy()
            reward_eval = reward_fn(torch.tensor(action_eval,device=device,dtype=torch.float32).unsqueeze(0), torch.tensor(next_obs_eval,device=device,dtype=torch.float32).unsqueeze(0)).squeeze().detach().cpu().numpy()
            eval_replay_buffer.add(obs_eval, action_eval, next_obs_eval, reward_eval, terminated_eval, truncated_eval)
            replay_buffer.add(obs_eval, action_eval, next_obs_eval, reward_eval, terminated_eval, truncated_eval)
            
            frame_eval = test_env.render()  # Get frame
            video_frames_eval.append(frame_eval)
            ReturnEval+=reward_eval
            obs_eval = next_obs_eval
            if terminated_eval or truncated_eval:
                if terminated_eval:
                    print("Terminated at Eval Episode:", eval_episode_counter)
                    countTerminated_eval+=1
                imageio.mimsave("cartpole.mp4", video_frames_eval, fps=30) 
                wandb.log({"PerformanceAgent-Eval/Video": wandb.Video("cartpole.mp4", fps=30, format="mp4")}, step = global_step)
                global_step += 1
                video_frames_eval=[]
        wandb.log({"PerformanceAgent-Eval/CountTerminated":countTerminated_eval,"PerformanceAgent-Eval/EvalReturn":ReturnEval/perf_real_env_evaluation_episodes, "PerformanceAgentTraining-Eval/TotalPerfEnvSteps":total_perf_env_steps}, step = global_step)

        if (ReturnEval/perf_real_env_evaluation_episodes)>maxReturnEval:
            maxReturnEval=ReturnEval/perf_real_env_evaluation_episodes
            best_filtered_agent_safety_filter = FastTD3Agent(dynasaur_model.observation_space,dynasaur_model.action_space_control, device, cfg, safety_networks, hyperplane_exploration=True, completeLoad=True)
            best_filtered_agent_perf_agent = FastTD3Agent(env.observation_space,env.action_space, device, cfg, fast_td3_agent_perf.get_td3_networks(), hyperplane_exploration=False, completeLoad=True)
            best_filtered_agent = FilteredAgent(best_filtered_agent_perf_agent,best_filtered_agent_safety_filter, device, dynasaur_model)
        # Retrain p̂_{j+1} on the expanded dataset D^{p,p̂}_j (Algorithm 2).
        # More data → larger certain set E_{j+1} → less restrictive µ_{j+1} next iteration.
        global_step = util.common.train_model_and_save_model_and_data(
                                                                global_step,
                                                                dynamics_model,
                                                                model_trainer,
                                                                cfg.algorithm,
                                                                replay_buffer,
                                                                work_dir=None,
                                                            )
        # Rebuild model wrapper with the retrained p̂_{j+1}, carrying over τ_G starting-state
        # history so the filter training distribution ρ^SF_0 remains informed across iterations.
        dynasaur_model_wrapper_new = dynasaurModelWrapper(model_env=model_env,replay_buffer=replay_buffer, cfg=cfg, env=env, device=device, starting_state_fn = starting_state_fn)
        dynasaur_model_wrapper_new.starting_states_reward_model_max_reward = dynasaur_model_wrapper.starting_states_reward_model_max_reward
        dynasaur_model_wrapper_new.starting_states_reward_model = dynasaur_model_wrapper.starting_states_reward_model
        dynasaur_model_wrapper_new.ground_truth_starting_states_reward = dynasaur_model_wrapper.ground_truth_starting_states_reward
        dynasaur_model_wrapper = dynasaur_model_wrapper_new
        dynasaur_model = HyperplaneSafetyWrapperEnv(dynasaur_model_wrapper, cfg, eval_mode = False)
        dynasaur_model_test = HyperplaneSafetyWrapperEnv(dynasaur_model_wrapper, cfg, eval_mode = True)
        if one_d_actionspace and one_d_extra_visuals:
            dynasaur_model.env.visualize_accurate_set(index_dynasaur=k, global_step=global_step, total_perf_env_steps=total_perf_env_steps)
        replay_buffer_obs_added = replay_buffer.obs[num_stored_old:replay_buffer.num_stored,:]
        if cfg.algorithm.completeSafe:
            replay_buffer_intermediate_added = util.common.create_replay_buffer_fixed_capacity(
                cfg,
                replay_buffer.num_stored-num_stored_old+1,
                obs_shape,
                act_shape,
                rng=rng,
                obs_type=dtype,
                action_type=dtype,
                reward_type=dtype,
            )
            replay_buffer_intermediate_added.add_batch(
                replay_buffer.obs[num_stored_old:replay_buffer.num_stored,:],
                replay_buffer.action[num_stored_old:replay_buffer.num_stored,:],
                replay_buffer.next_obs[num_stored_old:replay_buffer.num_stored,:],
                replay_buffer.reward[num_stored_old:replay_buffer.num_stored],
                replay_buffer.terminated[num_stored_old:replay_buffer.num_stored],
                replay_buffer.truncated[num_stored_old:replay_buffer.num_stored],
            )
            save_data_compressed(replay_buffer_intermediate_added, f"ReplayBufferAdded/RB-{k+1}.pkl")
            del replay_buffer_intermediate_added
            eval_replay_buffer_intermediate_added = util.common.create_replay_buffer_fixed_capacity(
                        cfg,
                        eval_replay_buffer.num_stored-num_stored_old_eval+1,
                        obs_shape,
                        act_shape,
                        rng=rng,
                        obs_type=dtype,
                        action_type=dtype,
                        reward_type=dtype,
                    )
            eval_replay_buffer_intermediate_added.add_batch(
                eval_replay_buffer.obs[num_stored_old_eval:eval_replay_buffer.num_stored,:],
                eval_replay_buffer.action[num_stored_old_eval:eval_replay_buffer.num_stored,:],
                eval_replay_buffer.next_obs[num_stored_old_eval:eval_replay_buffer.num_stored,:],
                eval_replay_buffer.reward[num_stored_old_eval:eval_replay_buffer.num_stored],
                eval_replay_buffer.terminated[num_stored_old_eval:eval_replay_buffer.num_stored],
                eval_replay_buffer.truncated[num_stored_old_eval:eval_replay_buffer.num_stored],
            )
            save_data_compressed(eval_replay_buffer_intermediate_added, f"EvalReplayBufferAdded/ERB-{k+1}.pkl")
            del eval_replay_buffer_intermediate_added
        replay_buffer_obs_all = replay_buffer.obs[:replay_buffer.num_stored,:]
        eval_replay_buffer_obs = eval_replay_buffer.obs
        plot_starting_states_ground_truth(replay_buffer_obs_all, replay_buffer_obs_added,eval_replay_buffer_obs, global_step,  cfg.algorithm.env_safety_args, total_perf_env_steps) if (debug_mode) else None
        # Inject τ_G states into the filter starting-state distribution ρ^SF_0 (ν3 term, eq. 20).
        if total_max_reward_starting_states is not None:
            dynasaur_model.env.update_starting_states(total_max_reward_starting_states, global_step=global_step, rewarding = True, total_steps=total_perf_env_steps, groundTruth=True)
        best_filtered_agent_perf_agent_networks = best_filtered_agent_perf_agent.get_td3_networks()
        if cfg.algorithm.completeSafe:
            save_data_compressed(dynamics_model, f"ModelCompleted/M-{k+1}.pkl")
            best_filtered_agent_perf_agent_networks.save(folder="PerformanceAgentCompleted", name=f"PA-{k+1}.pt")
        # Retrain filter µ_{j+1} on the updated model p̂_{j+1} (Algorithm 4/5).
        safety_networks, global_step,total_safety_env_steps = td3_safety_filter(dynasaur_model, dynasaur_model_test, cfg, standalone=False, old_networks = safety_networks ,completeLoad= cfg.algorithm.completeLoading, global_step_old = global_step, perf_agent=best_filtered_agent if cfg.algorithm.usePerfAgentProposal else None, real_env=env, total_safety_env_steps=total_safety_env_steps,  termination_fn_real=termination_fn, index_dynasor=k+1)
        if cfg.algorithm.completeSafe:
            safety_networks.save(folder="SafetyFilterCompleted", name=f"SF-{k+1}.pt")
        safety_filte_fast_td3_agent = FastTD3Agent(dynasaur_model.observation_space,dynasaur_model.action_space_control, device, cfg, safety_networks, hyperplane_exploration=True, completeLoad=True)
    if cfg.track_wandb:
        wandb.finish()
    return np.float32(0)
