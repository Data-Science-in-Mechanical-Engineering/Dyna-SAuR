import os
import numpy as np
from dynasaur.planning.sac_wrapper import SACAgent
import dynasaur.planning as planning
import numpy as np
import dynasaur.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
from dynasaur.util.data_management import save_data_compressed
import colorednoise as cn  
from tqdm import tqdm

def gather_data_walker(env, cfg, replay_buffer):
    planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(pytorch_sac_pranz24.SAC(cfg.algorithm.agent.num_inputs, env.action_space, cfg.algorithm.agent.args))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    absolute_path = os.path.join(current_dir, "../initialPriorController/sac_agent_walker.pth")
    absolute_path = os.path.normpath(absolute_path)  # Clean up the path
    agent.sac_agent.load_checkpoint(absolute_path)
    num_steps_policy = cfg.algorithm.prior_replaybuffer_size//2 #4Million Steps
    num_steps_uniform = cfg.algorithm.prior_replaybuffer_size//2

    steps = 0
    print("Starting prior data collection...")
    pbar = tqdm(total=num_steps_policy+num_steps_uniform, desc="Collecting data")
    while steps < num_steps_policy+num_steps_uniform:
        obs, info = env.reset()
        done = False
        pink_noise = cn.powerlaw_psd_gaussian(1, (env.action_space.shape[0],500)).T
        steps_episode = 0
        while not done:
            # Decide whether to use prior controller or random action
            action =agent.act(obs, sample=True)
            action = action.detach().cpu().numpy() + 0.2 * pink_noise[steps_episode]
            action = env.action_space.sample() if steps_episode%2==1 else action
            steps_episode +=1
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
            done = terminated or truncated
            obs = next_obs
            steps+=1
            pbar.update(1)
    pbar.close()
    print("Total prior data collected: "+str(replay_buffer.num_stored)+" steps")