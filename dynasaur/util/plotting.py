import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import wandb
import torch
import math
from cycler import cycler
from dynasaur.util.pink_noise import PinkNoiseGenerator
from dynasaur.algorithms.fast_td3_agent import FastTD3Agent
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import KDTree
def plotQ():
    pass

def plot_starting_states_ground_truth(replay_buffer_obs_all, replay_buffer_obs_added,eval_replay_buffer_obs, global_step, algorithm_safety_args, total_perf_env_steps):

    use_z_dim = algorithm_safety_args.use_third_visual_dim
    if use_z_dim:
        z_dim = algorithm_safety_args.index_third_visual_dim
        z_name = algorithm_safety_args.name_third_visual_dim
        z_lower = algorithm_safety_args.lower_lim_third_visual_dim
        z_upper = algorithm_safety_args.upper_lim_third_visual_dim
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim


    if use_z_dim:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        value_plot = ax.scatter(np.array(replay_buffer_obs_all)[:,z_dim], 
                       np.array(replay_buffer_obs_all)[:,y_dim],
                       np.array(replay_buffer_obs_all)[:,x_dim],
                       s=1, alpha=0.05)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title("Ground Truth Replay Buffer")
        plot_all = wandb.Image(plt)
    else:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        value_plot=plt.scatter(np.array(replay_buffer_obs_all)[:,x_dim], np.array(replay_buffer_obs_all)[:,y_dim])
        # This are the borders of the simulation  
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Ground Truth Replay Buffer")
        plot_all = wandb.Image(plt)

    if use_z_dim:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        value_plot = ax.scatter(np.array(eval_replay_buffer_obs)[:,z_dim], 
                       np.array(eval_replay_buffer_obs)[:,y_dim],
                       np.array(eval_replay_buffer_obs)[:,x_dim],
                       s=1, alpha=0.15)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title("Ground Truth Eval Replay Buffer")
        plot_eval = wandb.Image(plt)
    else:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        value_plot=plt.scatter(np.array(eval_replay_buffer_obs)[:,x_dim], np.array(eval_replay_buffer_obs)[:,y_dim])
        # This are the borders of the simulation  
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Ground Truth Eval Replay Buffer")
        plot_eval = wandb.Image(plt)
    if use_z_dim:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        value_plot = ax.scatter(np.array(replay_buffer_obs_added)[:,z_dim], 
                       np.array(replay_buffer_obs_added)[:,y_dim],
                       np.array(replay_buffer_obs_added)[:,x_dim],
                       s=1, alpha=0.15)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title("Ground Truth Replay Buffer Added")
        plot_added = wandb.Image(plt)
    else:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        value_plot=plt.scatter(np.array(replay_buffer_obs_added)[:,x_dim], np.array(replay_buffer_obs_added)[:,y_dim])
        # This are the borders of the simulation  
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Ground Truth Replay Buffer Added")
        plot_added = wandb.Image(plt)

    wandb.log({"Model_Learning-Data/GroundTruthReplayBuffer":plot_all,"Model_Learning-Data/GroundTruthReplayBufferAdded":plot_added, "Model_Learning-Data/GroundTruthReplayBufferAddedEval": plot_eval,
               "Model_Learning-Data/total_perf_env_steps": total_perf_env_steps},step=global_step)

def plot_pink_real(global_step, test_env, real_env, fast_td3_agent, seed, algorithm_safety_args, toal_safety_env_steps, pink_noise_action_generator, termination_fn=None, perf_agent=None, exploration_noise_real_env_perf=0):
    use_z_dim = algorithm_safety_args.use_third_visual_dim
    if use_z_dim:
        z_dim = algorithm_safety_args.index_third_visual_dim
        z_name = algorithm_safety_args.name_third_visual_dim
        z_lower = algorithm_safety_args.lower_lim_third_visual_dim
        z_upper = algorithm_safety_args.upper_lim_third_visual_dim
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    number_pink_noise_episodes  = algorithm_safety_args.number_real_pink_noise_episodes
    if perf_agent is None:
        obs_eval, _ = real_env.reset()
        episode_counter = 0
        trajectories_obs = []
        while episode_counter < number_pink_noise_episodes:
            traj_obs = []
            done = False
            truncated = False
            terminated = False
            i = 0
            pink_noise_action_generator.reset_pink()
            while not done:
                traj_obs.append(obs_eval)
                with torch.no_grad():
                    actions = fast_td3_agent.get_target_actor(torch.tensor(obs_eval, device=fast_td3_agent.device).unsqueeze(0).float())
                action_proposal = pink_noise_action_generator.get_pink_action()/3
                action_filtered, _ = test_env.filter_action_batch_penalty(actions, action_proposal)
                action_filtered = action_filtered.flatten().detach().cpu().numpy()
                next_obs, reward, _, truncated, infos = real_env.step(action_filtered)
                terminated = termination_fn(torch.tensor(action_filtered, device=fast_td3_agent.device).unsqueeze(0).float(), torch.tensor(next_obs, device=fast_td3_agent.device).unsqueeze(0).float()).item()
                obs_eval = next_obs
                done = terminated or truncated
                i+=1
            print("Finished pink noise episode", episode_counter, "with length", i, "!")
            traj_obs.append(obs_eval)
            traj_obs = np.array(traj_obs)
            trajectories_obs.append(traj_obs)
            episode_counter += 1
            obs_eval, _ = real_env.reset()
    else:
        obs_eval, _ = real_env.reset()
        episode_counter = 0
        trajectories_obs = []
        while episode_counter < number_pink_noise_episodes:
            traj_obs = []
            done = False
            truncated = False
            terminated = False
            i = 0
            pink_noise_action_generator.reset_pink()
            while not done:
                traj_obs.append(obs_eval)
                with torch.no_grad():
                    actions = fast_td3_agent.get_target_actor(torch.tensor(obs_eval, device=fast_td3_agent.device).unsqueeze(0).float())
                action_proposal = perf_agent.explore(torch.tensor(obs_eval, device=fast_td3_agent.device).unsqueeze(0).float(),eps_action=pink_noise_action_generator.get_pink_action(), noise_scales=torch.ones((1,), device=fast_td3_agent.device, dtype=torch.float32)*exploration_noise_real_env_perf) 
                action_filtered, _ = test_env.filter_action_batch_penalty(actions, action_proposal)
                action_filtered = action_filtered.flatten().detach().cpu().numpy()
                next_obs, reward, _, truncated, infos = real_env.step(action_filtered)
                terminated = termination_fn(torch.tensor(action_filtered, device=fast_td3_agent.device).unsqueeze(0).float(), torch.tensor(next_obs, device=fast_td3_agent.device).unsqueeze(0).float()).item()
                obs_eval = next_obs
                done = terminated or truncated
                i+=1
            print("Finished Perf pink noise episode", episode_counter, "with length", i, "!")
            traj_obs.append(obs_eval)
            traj_obs = np.array(traj_obs)
            trajectories_obs.append(traj_obs)
            episode_counter += 1
            obs_eval, _ = real_env.reset()
    # Plot trajectories
    if not use_z_dim:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, number_pink_noise_episodes))
        sum_episode_length_pink=0
        for i in range(number_pink_noise_episodes):
            traj = trajectories_obs[i]
            traj = traj.reshape(-1, traj.shape[-1])
            episode_length_pink = traj.shape[0] -1 #because of initial observation
            sum_episode_length_pink+=episode_length_pink
            ax.plot(traj[:-1,x_dim], traj[:-1,y_dim], color=colors[i], label=f'Eval {i+1}')
            ax.plot(traj[-1,x_dim], traj[-1,y_dim], color=colors[i] if episode_length_pink>=test_env.safe_episode_length else 'red', marker='o') 
        averageEpsidodeLengthPink = sum_episode_length_pink/number_pink_noise_episodes
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title('Trajectories with Pink Noise Action Proposal')
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        plt.tight_layout()
        fig.savefig("eval_trajectories_pink.png", dpi=200)
        plot4 = wandb.Image("eval_trajectories_pink.png")
    else:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, number_pink_noise_episodes))
        sum_episode_length_pink=0
        for i in range(number_pink_noise_episodes):
            traj = trajectories_obs[i]
            traj = traj.reshape(-1, traj.shape[-1])
            episode_length_pink = traj.shape[0] -1 #because of initial observation
            sum_episode_length_pink+=episode_length_pink
            ax.plot(traj[:-1,z_dim], traj[:-1,y_dim], traj[:-1,x_dim], color=colors[i], label=f'Eval {i+1}')
            marker_color = colors[i] if episode_length_pink>=test_env.safe_episode_length else 'red'
            ax.scatter(traj[-1,z_dim], traj[-1,y_dim], traj[-1,x_dim], color=marker_color, marker='o', s=50)
        
        averageEpsidodeLengthPink = sum_episode_length_pink/number_pink_noise_episodes
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title('Trajectories with Pink Noise Action Proposal')
        plt.tight_layout()
        fig.savefig("eval_trajectories_pink.png", dpi=200)
        plot4 = wandb.Image("eval_trajectories_pink.png")
    if perf_agent is None:
        wandb.log(data={
        "Safety_Filter_Learning-Pink/Trajectories_RealEnv_Pink": plot4,
        "Safety_Filter_Learning-Pink/AverageEpisodeLength_RealEnv_Pink": averageEpsidodeLengthPink,
        "Safety_Filter_Learning-Pink/TotalSafetyEnvSteps": toal_safety_env_steps,
        },step=global_step)
    else:
        wandb.log(data={
        "Safety_Filter_Learning-Pink/Trajectories_RealEnv_Perf": plot4,
        "Safety_Filter_Learning-Pink/AverageEpisodeLength_RealEnv_Perf": averageEpsidodeLengthPink,
        },step=global_step)
    return averageEpsidodeLengthPink

#Special for 1D action space only plotting function and right now only CartPole
def plot_hyperplanes(global_step, test_env, fast_td3_agent: FastTD3Agent, algorithm_safety_args, toal_safety_env_steps):
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    device = fast_td3_agent.device
    env = test_env
    plotsDecisions =[]
    plotsDecisionsLabel=[]
    for xdot,thetadot in [(-1,0),(-1,-1),(0,-1),(0,1),(-1,1),(1,-1),(1,0),(1,1), (0,0)]:
        borders = []
        arrowDirX=[]
        arrowDirY=[]
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        colors = []
        colors_v = []
        x_vals = np.arange(x_lower, x_upper+0.01, x_upper/20)
        theta_vals = np.arange(y_lower, y_upper+0.01, y_upper/20)
        
        # Initialize color array with white background (all 1s)
        color_array = np.ones((len(theta_vals), len(x_vals), 4))
        
        for i, theta in enumerate(theta_vals):
            for j, x in enumerate(x_vals):
                with torch.no_grad():
                    obs_eval = [[x, xdot, theta, thetadot]]
                    actions = fast_td3_agent.get_actor(torch.Tensor(obs_eval).to(device))
                    qf1_next_target = fast_td3_agent.get_qf1(torch.Tensor(obs_eval).to(device), actions)
                    qf2_next_target = fast_td3_agent.get_qf2(torch.Tensor(obs_eval).to(device), actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

                    actions = actions.cpu().numpy().clip(env.action_space.low[0], env.action_space.high[0])
                    
                    colors_v.append(min_qf_next_target.cpu().detach().numpy())
                    safety_action = actions
                    alpha = np.abs(safety_action).item()
                    if safety_action > 0:
                        color_safety = [102/255, 153/255, 204/255, alpha]  # blue for positive
                    else:
                        color_safety = [250/255, 144/255, 66/255, alpha]  # orange for negative
                    
                    color_array[i, j] = color_safety
                    borders.append([x, theta])

        borders = np.array(borders)

        # Change rectangle to high-contrast color
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, 
                                    linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rectangle)

        # Add colorbar to show action mapping
        from matplotlib.colors import LinearSegmentedColormap
        
        cmap = LinearSegmentedColormap.from_list('action_cmap', 
                                                ['#ff8c42', '#FFFFFF', '#6699cc'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        
        # Simplified labels
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['Drive Left\n[-1,-1]', '[-1,0]', 'All Safe\n[-1,1]', '[0,1]', 'Drive Right\n[1,1]'])
        cbar.set_label('Safe Action Interval', rotation=270, labelpad=25)

        ax.imshow(color_array, extent=[x_lower, x_upper, y_lower, y_upper], 
          origin='lower', aspect='auto', interpolation='nearest')

        plt.xlim(x_lower-0.1, x_upper+0.1)
        plt.ylim(y_lower-0.01, y_upper+0.01)
        plt.xlabel(x_name)
        plt.ylabel(r"$\theta$")
        plt.tight_layout() 
        fig.savefig(f"low_res_plot2{xdot}-{thetadot}.png", dpi=300)  # Reduce resolution
        plotsDecisions.append(wandb.Image(f"low_res_plot2{xdot}-{thetadot}.png") )
        plotsDecisionsLabel.append(f"Safety_Filter_Learning-HyperplanesOnline/SafetyFilterPolicyFor XDot:{xdot}+ThetaDot: {thetadot}")
        if xdot==0 and thetadot==0:
            #These will plot the value function of the ppo hyperplane learning
            fig = plt.figure(figsize=(8, 8), clear=True, num=1)
            ax = fig.add_subplot(111)
            rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
            ax.add_patch(rectangle)
            value_plot=plt.scatter(borders[:,0], borders[:,1], c= colors_v, cmap ="viridis", s=30)
            plt.colorbar(value_plot, label="Maximal Q Value Function")
            plt.clim(-100, 100)            
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.tight_layout()
            plt.title("Maximal Q Value Function")
            fig.savefig("low_res_plot3.png", dpi=100)  # Reduce resolution

    plotsDecisionsTarget =[]
    plotsDecisionsLabelTarget=[]
    for xdot,thetadot in [(-1,0),(-1,-1),(0,-1),(0,1),(-1,1),(1,-1),(1,0),(1,1), (0,0)]:
        borders = []
        arrowDirX=[]
        arrowDirY=[]
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        colors = []
        colors_v = []
        for x in np.arange(x_lower,x_upper+0.1, x_upper/20):
            for theta in np.arange(y_lower,y_upper,math.pi / 360 *4):
                with torch.no_grad():
                    obs_eval=[[x,xdot,theta,thetadot]]
                    actions = fast_td3_agent.get_actor(torch.Tensor(obs_eval).to(device))
                    qf1_next_target = fast_td3_agent.get_qf1(torch.Tensor(obs_eval).to(device), actions)
                    qf2_next_target = fast_td3_agent.get_qf2(torch.Tensor(obs_eval).to(device), actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

                    actions = actions.cpu().numpy().clip(env.action_space.low[0], env.action_space.high[0])
                    
                    colors_v.append(min_qf_next_target.cpu().detach().numpy())
                    borders.append([x,theta])
                    direction_h, threshold = env.double_to_hyperplane(actions)

                    if direction_h>0: # to left is dangerous
                        arrowDirX.append(0)
                        arrowDirY.append(1)
                        colors.append(threshold)
                    else: #to right is dangerous
                        arrowDirX.append(0)
                        arrowDirY.append(-1)
                        colors.append(-threshold)
        borders = np.array(borders)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        quiver_plot=plt.quiver(borders[:,0], borders[:,1], arrowDirX, arrowDirY,colors, cmap ="viridis", angles='xy', scale_units='xy', scale=50, width=0.003)
        # This are the learned hyperplanes plottted in state space
        plt.colorbar(quiver_plot, label="Safe actions from value in direction of arrow")  
        plt.clim(-1, 1)          
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.tight_layout()
        plt.title(f"TargetSafetyFilterPolicyFor Xdot:{xdot}+Thetadot:{thetadot}")
        fig.savefig(f"low_res_plot7{xdot}-{thetadot}.png", dpi=100)  # Reduce resolution
        plotsDecisionsTarget.append(wandb.Image(f"low_res_plot7{xdot}-{thetadot}.png") )
        plotsDecisionsLabelTarget.append(f"Safety_Filter_Learning-HyperplanesTarget/SafetyFilterPolicyFor XDot:{xdot}+ThetaDot: {thetadot}")
        if xdot==0 and thetadot==0:
            #These will plot the value function of the ppo hyperplane learning
            fig = plt.figure(figsize=(8, 8), clear=True, num=1)
            ax = fig.add_subplot(111)
            rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
            ax.add_patch(rectangle)
            value_plot=plt.scatter(borders[:,0], borders[:,1], c= colors_v, cmap ="viridis", s=30)
            plt.colorbar(value_plot, label="Maximal Q Value Function Target")   
            plt.clim(-100, 100)         
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title("Maximal Q Value Function Target")
            fig.savefig("low_res_plot37.png", dpi=100)  # Reduce resolution

    plot3 = wandb.Image("low_res_plot3.png") 
    plot37 = wandb.Image("low_res_plot37.png") 
    wandb.log(data={"Safety_Filter_Learning-HyperplanesOnline/global_step": global_step,
        "Safety_Filter_Learning-HyperplanesOnline/TotalSafetyEnvSteps": toal_safety_env_steps,
        "Safety_Filter_Learning-HyperplanesOnline/ValueFunction": plot3,
        "Safety_Filter_Learning-HyperplanesTarget/TargetValueFunction": plot37,
        **{f"{key}": plot for key, plot in zip(plotsDecisionsLabel, plotsDecisions)},
        **{f"{key}": plot for key, plot in zip(plotsDecisionsLabelTarget, plotsDecisionsTarget)}
        },
            step=global_step)

#1D action space only plotting function
def plot_explorative_trajectories(env, trajectories_training_action, trajectories_training, global_step, algorithm_safety_args,toal_safety_env_steps):
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    trajectories_training = trajectories_training.reshape(-1, trajectories_training.shape[-1])
    trajectories_training_action = trajectories_training_action.reshape(-1, trajectories_training_action.shape[-1])
    direction_h, threshold = env.double_to_hyperplane_batch(trajectories_training_action)
    fig = plt.figure(figsize=(8, 8), clear=True, num=1)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    plt.scatter(trajectories_training[:,x_dim], trajectories_training[:,y_dim])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    ax.set_title('Trajectories')
    plt.tight_layout()
    fig.savefig("eval_trajectories_2.png", dpi=200)

    fig = plt.figure(figsize=(8, 8), clear=True, num=1)
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    plt.quiver(trajectories_training[:,x_dim], trajectories_training[:,y_dim], np.zeros_like(direction_h),direction_h, threshold/np.linalg.norm(direction_h), cmap ="viridis", angles='xy', scale_units='xy', scale=20, width=0.003)
    ax.plot()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    ax.set_title('HyperplaneExploration ')
    plt.tight_layout()
    fig.savefig("eval_trajectories_3.png", dpi=200)
    wandb.log({
        "Safety_Filter_Learning-Training/TotalSafetyEnvSteps": toal_safety_env_steps,
        "Safety_Filter_Learning-Training/HyperplaneExploration": wandb.Image("eval_trajectories_3.png"),
        "Safety_Filter_Learning-Training/Trajectory": wandb.Image("eval_trajectories_2.png"),
    }, step = global_step)




def plot_eval_trajectories(global_step, trajectory_q_value, trajectories_obs,dict_trajectory_terminated, dict_trajectory_truncated,  algorithm_safety_args, trial_legnth, toal_safety_env_steps, perfAgentNotPlotted, prefix="", eval_episodes=0, safetyQThreshold=50):
    use_z_dim = algorithm_safety_args.use_third_visual_dim
    if use_z_dim:
        z_dim = algorithm_safety_args.index_third_visual_dim
        z_name = algorithm_safety_args.name_third_visual_dim
        z_lower = algorithm_safety_args.lower_lim_third_visual_dim
        z_upper = algorithm_safety_args.upper_lim_third_visual_dim
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    if perfAgentNotPlotted==True:
        # Plot only pink trajectories
        trajectories_pink = trajectories_obs
        trajectory_q_value_pink = trajectory_q_value
        trajectories_perf = None
        trajectory_q_value_perf = None
        fig_empty = plt.figure(figsize=(8, 8), clear=True, num=2)
        ax_empty = fig_empty.add_subplot(111)
        ax_empty.text(0.5, 0.5, 'No Performance Trajectories', ha='center', va='center')
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        fig_empty.savefig("eval_trajectories_perf.png", dpi=200)
        fig_empty = plt.figure(figsize=(8, 8), clear=True, num=2)
        ax_empty = fig_empty.add_subplot(111)
        ax_empty.text(0.5, 0.5, 'No Performance Trajectories', ha='center', va='center')
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])
        fig_empty.savefig("safe_eval_trajectories_perf.png", dpi=200)
    else:
        trajectories_pink = trajectories_obs[:, :eval_episodes, :]
        trajectories_perf = trajectories_obs[:, eval_episodes:, :]
        trajectory_q_value_pink = trajectory_q_value[:, :eval_episodes]
        trajectory_q_value_perf = trajectory_q_value[:, :eval_episodes:]
    if not use_z_dim:
        #############################################2D Pink Eval Safe Rewarding Plot###############################################   
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        batch_size = trajectories_pink.shape[1]
        T = trajectories_pink.shape[0]
        colors = np.array([[0, 0, 0, 1]] * batch_size)
        # build terminated/truncated arrays (vectorized)
        term_times = np.fromiter((dict_trajectory_terminated.get(i, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
        trunc_times = np.fromiter((dict_trajectory_truncated.get(i, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)

        # decide end indices per batch element (cap to trajectory length)
        end_idx = np.minimum(np.minimum(term_times, trunc_times), T)
        
        # Vectorized computation of safe trajectory end indices
        traj_end_safe = np.ones(batch_size, dtype=int)
        for i in range(batch_size):
            if end_idx[i] > 1:
                safe_mask = trajectory_q_value_pink[:end_idx[i]-1, i] > safetyQThreshold
                if np.any(safe_mask):
                    traj_end_safe[i] = np.max(np.where(safe_mask)[0])
        
        traj_end_safe = np.minimum(end_idx, traj_end_safe)

        both_500_mask = (term_times == trial_legnth) & (trunc_times == trial_legnth)

        # mask indicating which (time, batch) slots should be plotted as line points
        mask_lines = (np.arange(T)[:, None] < (traj_end_safe - 1)[None, :])
        # extract x and theta for all time steps up to T-1
        x_all = trajectories_pink[:, :, x_dim]
        theta_all = trajectories_pink[:, :, y_dim]
        # place NaN where points should not be plotted
        X_lines = np.where(mask_lines, x_all, np.nan)
        Y_lines = np.where(mask_lines, theta_all, np.nan)
        Colors_lines = np.where(mask_lines[:-1], trajectory_q_value_pink, np.nan)
        # use prop cycle to assign per-column colors and plot all columns at once
        ax.set_prop_cycle(cycler('color', [tuple(c) for c in colors]))
        ax.plot(X_lines, Y_lines)

        sc = ax.scatter(X_lines[:-1], 
                Y_lines[:-1],
                c=Colors_lines,
                zorder=10,
                cmap='viridis',
                vmin=-100,
                vmax=100)
        
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('Q-Value')
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title('Evaluation Trajectories Safe(Pink)')
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        plt.tight_layout()
        fig.savefig("eval_trajectorie_safe_pink.png", dpi=200)
        #############################################2D Pink Eval Plot###############################################
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)

        # mask indicating which (time, batch) slots should be plotted as line points
        mask_lines = (np.arange(T)[:, None] < (end_idx - 1)[None, :])
        # place NaN where points should not be plotted
        X_lines = np.where(mask_lines, x_all, np.nan)
        Y_lines = np.where(mask_lines, theta_all, np.nan)
        Colors_lines = np.where(mask_lines[:-1], trajectory_q_value_pink, np.nan)
        # use prop cycle to assign per-column colors and plot all columns at once
        ax.set_prop_cycle(cycler('color', [tuple(c) for c in colors]))
        ax.plot(X_lines, Y_lines)

        sc = ax.scatter(X_lines[:-1], 
                Y_lines[:-1],
                c=Colors_lines,
                zorder=10,
                cmap='viridis',
                vmin=-100,
                vmax=100)
        
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('Q-Value')

        # Markers
        event_mask = ~both_500_mask
        valid_marker_mask = event_mask & (end_idx > 0)
        valid_inds = np.nonzero(valid_marker_mask)[0]

        if valid_inds.size > 0:
            last_indices = end_idx[valid_inds] - 1
            last_x = trajectories_pink[last_indices, valid_inds, x_dim]
            last_theta = trajectories_pink[last_indices, valid_inds, y_dim]

            terminated_before = (term_times <= trunc_times) & (term_times != trial_legnth)
            color_obj = np.array([tuple(c) for c in colors], dtype=object)
            marker_colors_all = np.array(['red' if terminated_before[i] else color_obj[i] for i in range(len(terminated_before))], dtype=object)
            marker_colors = marker_colors_all[valid_inds]

            ax.scatter(last_x, last_theta, c=marker_colors, marker='s', zorder=10)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title('Evaluation Trajectories (Pink)')
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        plt.tight_layout()
        fig.savefig("eval_trajectories.png", dpi=200)
        if perfAgentNotPlotted==False:
           #############################################Perf Agent 2D Perf Eval Plot Safe Trajectories Perf###############################################
            fig = plt.figure(figsize=(8, 8), clear=True, num=1)
            ax = fig.add_subplot(111)
            batch_size = trajectories_perf.shape[1]
            T = trajectories_perf.shape[0]
            colors = np.array([[0, 0, 0, 1]] * batch_size)
            # build terminated/truncated arrays (vectorized)
            term_times = np.fromiter((dict_trajectory_terminated.get(i+eval_episodes, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
            trunc_times = np.fromiter((dict_trajectory_truncated.get(i+eval_episodes, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)

            # decide end indices per batch element (cap to trajectory length)
            end_idx = np.minimum(np.minimum(term_times, trunc_times), T)
            # Vectorized computation of safe trajectory end indices
            traj_end_safe = np.ones(batch_size, dtype=int)
            for i in range(batch_size):
                if end_idx[i] > 1:
                    safe_mask = trajectory_q_value[:end_idx[i]-1, i] > safetyQThreshold
                    if np.any(safe_mask):
                        traj_end_safe[i] = np.max(np.where(safe_mask)[0])
            
            traj_end_safe = np.minimum(end_idx, traj_end_safe)

            both_500_mask = (term_times == trial_legnth) & (trunc_times == trial_legnth)

            # mask indicating which (time, batch) slots should be plotted as line points
            mask_lines = (np.arange(T)[:, None] < (traj_end_safe - 1)[None, :])
            # extract x and theta for all time steps up to T-1
            x_all = trajectories_perf[:, :, x_dim]
            theta_all = trajectories_perf[:, :, y_dim]
            # place NaN where points should not be plotted
            X_lines = np.where(mask_lines, x_all, np.nan)
            Y_lines = np.where(mask_lines, theta_all, np.nan)
            Colors_lines = np.where(mask_lines[:-1], trajectory_q_value_perf, np.nan)
            # use prop cycle to assign per-column colors and plot all columns at once
            ax.set_prop_cycle(cycler('color', [tuple(c) for c in colors]))
            ax.plot(X_lines, Y_lines)

            sc = ax.scatter(X_lines[:-1], 
                    Y_lines[:-1],
                    c=Colors_lines,
                    zorder=10,
                    cmap='viridis',
                    vmin=-100,
                    vmax=100)
            
            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Q-Value')

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title('Evaluation Trajectories (Perf)')
            rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
            ax.add_patch(rectangle)
            plt.tight_layout()
            fig.savefig("safe_eval_trajectories_perf.png", dpi=200)
            #############################################Perf Agent 2D Perf Eval Plot###############################################
            fig = plt.figure(figsize=(8, 8), clear=True, num=1)
            ax = fig.add_subplot(111)

            # mask indicating which (time, batch) slots should be plotted as line points
            mask_lines = (np.arange(T)[:, None] < (end_idx - 1)[None, :])
            # place NaN where points should not be plotted
            X_lines = np.where(mask_lines, x_all, np.nan)
            Y_lines = np.where(mask_lines, theta_all, np.nan)
            Colors_lines = np.where(mask_lines[:-1], trajectory_q_value_perf, np.nan)
            # use prop cycle to assign per-column colors and plot all columns at once
            ax.set_prop_cycle(cycler('color', [tuple(c) for c in colors]))
            ax.plot(X_lines, Y_lines)

            sc = ax.scatter(X_lines[:-1], 
                    Y_lines[:-1],
                    c=Colors_lines,
                    zorder=10,
                    cmap='viridis',
                    vmin=-100,
                    vmax=100)
            
            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Q-Value')

            # Markers
            event_mask = ~both_500_mask
            valid_marker_mask = event_mask & (end_idx > 0)
            valid_inds = np.nonzero(valid_marker_mask)[0]

            if valid_inds.size > 0:
                last_indices = end_idx[valid_inds] - 1
                last_x = trajectories_perf[last_indices, valid_inds, x_dim]
                last_theta = trajectories_perf[last_indices, valid_inds, y_dim]

                terminated_before = (term_times <= trunc_times) & (term_times != trial_legnth)
                color_obj = np.array([tuple(c) for c in colors], dtype=object)
                marker_colors_all = np.array(['red' if terminated_before[i] else color_obj[i] for i in range(len(terminated_before))], dtype=object)
                marker_colors = marker_colors_all[valid_inds]

                ax.scatter(last_x, last_theta, c=marker_colors, marker='s', zorder=10)

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title('Evaluation Trajectories (Perf)')
            rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
            ax.add_patch(rectangle)
            plt.tight_layout()
            fig.savefig("eval_trajectories_perf.png", dpi=200)
    else:
        ########################################################3D Plotting###############################################
        #############################################No Perf Agent 3D Pink Eval Safe Rewarding Plot###############################################
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        
        batch_size = trajectories_pink.shape[1]
        T = trajectories_pink.shape[0]
        colors = np.array([[0, 0, 0, 1]] * batch_size)
        
        term_times = np.fromiter((dict_trajectory_terminated.get(i, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
        trunc_times = np.fromiter((dict_trajectory_truncated.get(i, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
        
        end_idx = np.minimum(np.minimum(term_times, trunc_times), T)
        both_500_mask = (term_times == trial_legnth) & (trunc_times == trial_legnth)
        sc = None
        indices_plotted = range(batch_size) 
        for i in indices_plotted:
            traj_end = end_idx[i]
            traj_end_safe = np.where(trajectory_q_value_pink[:traj_end-1,i]>safetyQThreshold)
            traj_end_safe = np.max(traj_end_safe) if traj_end_safe[0].size>0 else 1
            traj_end = min(traj_end, traj_end_safe)
            if traj_end > 1:
                ax.plot(trajectories_pink[:traj_end-1, i, z_dim], 
                        trajectories_pink[:traj_end-1, i, y_dim],
                        trajectories_pink[:traj_end-1, i, x_dim],
                        color=colors[i])
                sc = ax.scatter(trajectories_pink[:traj_end-1, i, z_dim], 
                        trajectories_pink[:traj_end-1, i, y_dim],
                        trajectories_pink[:traj_end-1, i, x_dim],
                        c=trajectory_q_value[:traj_end-1,i],
                        cmap='viridis',
                        vmin=-100,
                        vmax=100)
                ax.scatter(trajectories_pink[traj_end-1, i, z_dim], trajectories_pink[traj_end-1, i, y_dim], trajectories_pink[traj_end-1, i, x_dim], c="grey", marker='s', s=50, zorder=10)
        
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('Q-Value')
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title('Safe Rewarding Trajectories (Pink)')
        plt.tight_layout()
        fig.savefig("eval_trajectorie_safe_pink.png", dpi=200)

        #############################################No Perf Agent 3D Pink Plot###############################################

        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        
        sc = None
        indices_plotted = range(batch_size) 
        for i in indices_plotted:
            traj_end = end_idx[i]
            if traj_end > 1:
                ax.plot(trajectories_pink[:traj_end-1, i, z_dim], 
                        trajectories_pink[:traj_end-1, i, y_dim],
                        trajectories_pink[:traj_end-1, i, x_dim],
                        color=colors[i])
                sc = ax.scatter(trajectories_pink[:traj_end-1, i, z_dim], 
                        trajectories_pink[:traj_end-1, i, y_dim],
                        trajectories_pink[:traj_end-1, i, x_dim],
                        c=trajectory_q_value_pink[:traj_end-1,i],
                        cmap='viridis',
                        vmin=-100,
                        vmax=100)
                ax.scatter(trajectories_pink[traj_end-1, i, z_dim], trajectories_pink[traj_end-1, i, y_dim], trajectories_pink[traj_end-1, i, x_dim], c="grey", marker='s', s=50, zorder=10)
        
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('Q-Value')
        
        event_mask = ~both_500_mask
        valid_marker_mask = event_mask & (end_idx > 0)
        valid_inds = np.nonzero(valid_marker_mask)[0]
        
        if valid_inds.size > 0:
            last_indices = end_idx[valid_inds] - 1
            last_z = trajectories_pink[last_indices, valid_inds, z_dim]
            last_y = trajectories_pink[last_indices, valid_inds, y_dim]
            last_x = trajectories_pink[last_indices, valid_inds, x_dim]
            
            terminated_before = (term_times <= trunc_times) & (term_times != trial_legnth)
            color_obj = np.array([tuple(c) for c in colors], dtype=object)
            marker_colors_all = np.array(['red' if terminated_before[i] else color_obj[i] for i in range(len(terminated_before))], dtype=object)
            marker_colors = marker_colors_all[valid_inds]
            
            ax.scatter(last_z, last_y, last_x, c=marker_colors, marker='s', s=50, zorder=10)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title('Evaluation Trajectories (Pink)')
        plt.tight_layout()
        fig.savefig("eval_trajectories.png", dpi=200)   
        if perfAgentNotPlotted==False:
        #############################################Perf Agent 3D Plot Safe Trajectories Perf###############################################
            fig = plt.figure(figsize=(10, 10), clear=True, num=1)
            ax = fig.add_subplot(111, projection='3d')
            batch_size = trajectories_perf.shape[1]
            T = trajectories_perf.shape[0]
            colors = np.array([[0, 0, 0, 1]] * batch_size)
            
            term_times = np.fromiter((dict_trajectory_terminated.get(i+eval_episodes, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
            trunc_times = np.fromiter((dict_trajectory_truncated.get(i+eval_episodes, trial_legnth) for i in range(batch_size)), dtype=int, count=batch_size)
            
            end_idx = np.minimum(np.minimum(term_times, trunc_times), T)
            both_500_mask = (term_times == trial_legnth) & (trunc_times == trial_legnth)
            
            sc = None
            for i in range(batch_size):
                traj_end = end_idx[i]
                traj_end_safe = np.where(trajectory_q_value_pink[:traj_end-1,i]>safetyQThreshold)
                traj_end_safe = np.max(traj_end_safe) if traj_end_safe[0].size>0 else 1
                traj_end = min(traj_end, traj_end_safe)
                if traj_end > 1:
                    ax.plot(trajectories_perf[:traj_end-1, i, z_dim], 
                           trajectories_perf[:traj_end-1, i, y_dim],
                           trajectories_perf[:traj_end-1, i, x_dim],
                           color=colors[i])
                    sc = ax.scatter(trajectories_perf[:traj_end-1, i, z_dim], 
                           trajectories_perf[:traj_end-1, i, y_dim],
                           trajectories_perf[:traj_end-1, i, x_dim],
                           c=trajectory_q_value_perf[:traj_end-1,i],
                           cmap='viridis',
                           vmin=-100,
                           vmax=100)
                    ax.scatter(trajectories_perf[traj_end-1, i, z_dim], trajectories_perf[traj_end-1, i, y_dim], trajectories_perf[traj_end-1, i, x_dim], c="grey", marker='s', s=50, zorder=10)
            
            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Q-Value')
            
            ax.set_xlabel(z_name)
            ax.set_ylabel(y_name)
            ax.set_zlabel(x_name)
            ax.set_title('Evaluation Trajectories (Perf)')
            plt.tight_layout()
            fig.savefig("safe_eval_trajectories_perf.png", dpi=200)
            #############################################Perf Agent 3D Perf Eval Pink Plot###############################################
            fig = plt.figure(figsize=(10, 10), clear=True, num=1)
            ax = fig.add_subplot(111, projection='3d')
            sc = None
            for i in range(batch_size):
                traj_end = end_idx[i]
                if traj_end > 1:
                    ax.plot(trajectories_perf[:traj_end-1, i, z_dim], 
                           trajectories_perf[:traj_end-1, i, y_dim],
                           trajectories_perf[:traj_end-1, i, x_dim],
                           color=colors[i])
                    sc = ax.scatter(trajectories_perf[:traj_end-1, i, z_dim], 
                           trajectories_perf[:traj_end-1, i, y_dim],
                           trajectories_perf[:traj_end-1, i, x_dim],
                           c=trajectory_q_value_perf[:traj_end-1,i],
                           cmap='viridis',
                           vmin=-100,
                           vmax=100)
            
            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Q-Value')
            
            event_mask = ~both_500_mask
            valid_marker_mask = event_mask & (end_idx > 0)
            valid_inds = np.nonzero(valid_marker_mask)[0]
            
            if valid_inds.size > 0:
                last_indices = end_idx[valid_inds] - 1
                last_z = trajectories_perf[last_indices, valid_inds, z_dim]
                last_y = trajectories_perf[last_indices, valid_inds, y_dim]
                last_x = trajectories_perf[last_indices, valid_inds, x_dim]
                
                terminated_before = (term_times <= trunc_times) & (term_times != trial_legnth)
                color_obj = np.array([tuple(c) for c in colors], dtype=object)
                marker_colors_all = np.array(['red' if terminated_before[i] else color_obj[i] for i in range(len(terminated_before))], dtype=object)
                marker_colors = marker_colors_all[valid_inds]
                
                ax.scatter(last_z, last_y, last_x, c=marker_colors, marker='s', s=50, zorder=10)
            
            ax.set_xlabel(z_name)
            ax.set_ylabel(y_name)
            ax.set_zlabel(x_name)
            ax.set_title('Evaluation Trajectories (Perf)')
            plt.tight_layout()
            fig.savefig("eval_trajectories_perf.png", dpi=200)
    wandb.log({f"Safety_Filter_Learning-Evaluation/TrajectoriesPink-{prefix}": wandb.Image("eval_trajectories.png"),
            f"Safety_Filter_Learning-Evaluation/TrajectoriesPerf-{prefix}": wandb.Image("eval_trajectories_perf.png"),
            f"Safety_Filter_Learning-Evaluation/TrajectoriesPinkSafe-{prefix}": wandb.Image("eval_trajectorie_safe_pink.png"),
            f"Safety_Filter_Learning-Evaluation/TrajectoriesPerfSafe-{prefix}": wandb.Image("safe_eval_trajectories_perf.png"),
            "Safety_Filter_Learning-Evaluation/toal_safety_env_steps": toal_safety_env_steps
            }, step=global_step)

def plot_starting_states_failure(starting_states_failures, global_step, algorithm_safety_args, toal_safety_env_steps):
    use_z_dim = algorithm_safety_args.use_third_visual_dim
    if use_z_dim:
        z_dim = algorithm_safety_args.index_third_visual_dim
        z_name = algorithm_safety_args.name_third_visual_dim
        z_lower = algorithm_safety_args.lower_lim_third_visual_dim
        z_upper = algorithm_safety_args.upper_lim_third_visual_dim
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    if not use_z_dim:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        value_plot=plt.scatter(starting_states_failures[:,x_dim], starting_states_failures[:,y_dim])
        # This are the borders of the simulation
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Starting States Failure")
        fig.savefig("low_res_plot2466.png", dpi=150)  # Reduce resolution
        plot24 = wandb.Image("low_res_plot2466.png") 
    else:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        
        value_plot = ax.scatter(starting_states_failures[:,z_dim], 
                       starting_states_failures[:,y_dim],
                       starting_states_failures[:,x_dim],
                       s=30, alpha=0.8)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title("Starting States Failure")
        plt.tight_layout()
        fig.savefig("low_res_plot2466.png", dpi=150)
        plot24 = wandb.Image("low_res_plot2466.png")
    wandb.log({"Safety_Filter_Learning-Pink/FailureStartingStates":plot24,
               "Safety_Filter_Learning-Pink/toal_safety_env_steps": toal_safety_env_steps}, step = global_step)

def plot_starting_states_reward(starting_states, global_step, algorithm_safety_args, toal_perf_env_steps):
    use_z_dim = algorithm_safety_args.use_third_visual_dim
    if use_z_dim:
        z_dim = algorithm_safety_args.index_third_visual_dim
        z_name = algorithm_safety_args.name_third_visual_dim
        z_lower = algorithm_safety_args.lower_lim_third_visual_dim
        z_upper = algorithm_safety_args.upper_lim_third_visual_dim
    x_dim = algorithm_safety_args.index_first_visual_dim
    y_dim = algorithm_safety_args.index_second_visual_dim
    x_lower = algorithm_safety_args.lower_lim_first_visual_dim
    x_upper = algorithm_safety_args.upper_lim_first_visual_dim
    y_lower = algorithm_safety_args.lower_lim_second_visual_dim
    y_upper = algorithm_safety_args.upper_lim_second_visual_dim
    x_name = algorithm_safety_args.name_first_visual_dim
    y_name = algorithm_safety_args.name_second_visual_dim
    if not use_z_dim:
        fig = plt.figure(figsize=(8, 8), clear=True, num=1)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        value_plot=plt.scatter(starting_states[:,x_dim], starting_states[:,y_dim])
        # This are the borders of the simulation     
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Starting States Reward")
        fig.savefig("low_res_plot24.png", dpi=150)  # Reduce resolution
        plot24 = wandb.Image("low_res_plot24.png") 
    else:
        fig = plt.figure(figsize=(10, 10), clear=True, num=1)
        ax = fig.add_subplot(111, projection='3d')
        value_plot = ax.scatter(starting_states[:,z_dim], 
                       starting_states[:,y_dim],
                       starting_states[:,x_dim],
                       s=30, alpha=0.8)
        
        ax.set_xlabel(z_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(x_name)
        ax.set_title("Starting States Reward")
        plt.tight_layout()
        fig.savefig("low_res_plot24.png", dpi=150)
        plot24 = wandb.Image("low_res_plot24.png")
    wandb.log({"Safety_Filter_Learning-Pink/RewardingStartingStates":plot24,
              # "Safety_Filter_Learning-Pink/toal_perf_env_steps": toal_perf_env_steps
               }, step = global_step)
    
def define_metrics(obs_shape):
    wandb.define_metric("PerformanceAgentTraining-Train/sac_step")
    wandb.define_metric("PerformanceAgentTraining-Train/Video", step_metric="sac_step")
    wandb.define_metric("PerformanceAgentTraining-Train/CountTerminated", step_metric="sac_step")
    wandb.define_metric("PerformanceAgentTraining-Train/PerfActions", step_metric="sac_step")
    wandb.define_metric("PerformanceAgentTraining-Train/EpisodeCounterEvalTerminationCondtion", step_metric="sac_step")
    wandb.define_metric("PerformanceAgentTraining-Train/AverageReturnDuringSACTraining", step_metric="sac_step")
    wandb.define_metric("PerformanceAgentTraining-Losses/updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/batch_reward", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/actor_loss", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/actor_target_entropy", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/critic_loss", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/alpha_loss", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/alpha_value", step_metric="updates_made_sac")
    wandb.define_metric("PerformanceAgentTraining-Losses/actor_entropy", step_metric="updates_made_sac")



def visualize_accurately_modelled_set(x_grid_plot, z_grid_plot, difference_accuaracy_real, average_lost_info,average_lost_K,per_step_epistemic_frac, global_step, algorithm_safety_args, toal_perf_env_steps):
        per_step_epistemic_frac = per_step_epistemic_frac.detach().cpu().numpy().squeeze()
        x_dim = algorithm_safety_args.index_first_visual_dim
        y_dim = algorithm_safety_args.index_second_visual_dim
        x_lower = algorithm_safety_args.lower_lim_first_visual_dim
        x_upper = algorithm_safety_args.upper_lim_first_visual_dim
        y_lower = algorithm_safety_args.lower_lim_second_visual_dim
        y_upper = algorithm_safety_args.upper_lim_second_visual_dim
        x_name = algorithm_safety_args.name_first_visual_dim
        y_name = algorithm_safety_args.name_second_visual_dim
        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=difference_accuaracy_real, cmap="viridis", vmin = 0, vmax=0.1)
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Euclidian Distance")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Suifficient Model Accuaracy Euclidian Distance")
        
        # Show the plot
        img5 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=difference_accuaracy_real<0.01, cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Euclidian Distance")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Suifficient Model Accuaracy Euclidian Distance <0.01")
        # Show the plot
        img6 = wandb.Image(plt)                
        
        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=difference_accuaracy_real<0.02, cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Euclidian Distance")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Suifficient Model Accuaracy Euclidian Distance <0.02")
        # Show the plot
        img7 = wandb.Image(plt)  

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=difference_accuaracy_real<0.015, cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Euclidian Distance")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Suifficient Model Accuaracy Euclidian Distance <0.015")
        # Show the plot
        img8 = wandb.Image(plt)  
        
        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_info[:,0], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Infoloss")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) X Infoloss")
        # Show the plot
        img = wandb.Image(plt)    
        
        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_info[:,2], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Infoloss")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Theta Infoloss")
        # Show the plot
        img2 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_info[:,1], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Infoloss")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Xdot Infoloss")
        # Show the plot
        img3 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_info[:,3], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="Infoloss")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Thetadot Infoloss")
        # Show the plot
        img4 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,0], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) X (K)")
        # Show the plot
        img9 = wandb.Image(plt)    

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,2], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Theta (K)")
        # Show the plot
        img10 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,1], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) XDot (K)")
        # Show the plot
        img11 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,3], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Thetadot (K)")
        # Show the plot
        img12 = wandb.Image(plt)

        # Log what is certain and what not according to current threshold

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,0]<per_step_epistemic_frac[0], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) X (K)")
        # Show the plot
        img13 = wandb.Image(plt)    

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,2]<per_step_epistemic_frac[2], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Theta (K)")
        # Show the plot
        img14 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,1]<per_step_epistemic_frac[1], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) XDot (K)")
        # Show the plot
        img15 = wandb.Image(plt)

        fig = plt.figure(figsize=(8, 8), clear=True, num=0)
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, linewidth=2, edgecolor='green', facecolor='white')
        ax.add_patch(rectangle)
        scatter_plot = ax.scatter(x_grid_plot.flatten(), z_grid_plot.flatten(), c=average_lost_K[:,3]<per_step_epistemic_frac[3], cmap="viridis")
        # Set the plot boundaries
        plt.tight_layout()
        plt.colorbar(scatter_plot, label="(K)")
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Average(xdot,thetadot=-0.25,0,0.25) Thetadot (K)")
        # Show the plot
        img16 = wandb.Image(plt)

        wandb.log({
                    "Model_Learning-Quality/Average (K) X Thresholded":img13,
                    "Model_Learning-Quality/Average (K) Theta Thresholded":img14,
                    "Model_Learning-Quality/Average (K) XDot Thresholded":img15,
                    "Model_Learning-Quality/Average (K) ThetadDot Thresholded":img16,
                    "Model_Learning-Quality/Average (K) X":img9,
                    "Model_Learning-Quality/Average (K) Theta":img10,
                    "Model_Learning-Quality/Average (K) XDot":img11,
                    "Model_Learning-Quality/Average (K) ThetadDot":img12,
                    "Model_Learning-Quality/Average Infoloss X":img,
                    "Model_Learning-Quality/Average Infoloss Theta":img2,
                    "Model_Learning-Quality/Average Infoloss XDot":img3,
                    "Model_Learning-Quality/Average Infoloss ThetadDot":img4, 
                    "Model_Learning-Quality/EuclidianAccuracy":img5,
                    "Model_Learning-Quality/EuclidianAccuracy<0.01":img6,
                    "Model_Learning-Quality/EuclidianAccuracy<0.02":img7, 
                    "Model_Learning-Quality/EuclidianAccuracy<0.015":img8,
                    "Model_Learning-Quality/per_step_epistemic_frac_x":per_step_epistemic_frac[0],
                    "Model_Learning-Quality/per_step_epistemic_frac_xdot":per_step_epistemic_frac[1],
                    "Model_Learning-Quality/per_step_epistemic_frac_theta":per_step_epistemic_frac[2],
                    "Model_Learning-Quality/per_step_epistemic_frac_thetadot":per_step_epistemic_frac[3],
                    "Model_Learning-Quality/toal_perf_env_steps":toal_perf_env_steps}, step = global_step)