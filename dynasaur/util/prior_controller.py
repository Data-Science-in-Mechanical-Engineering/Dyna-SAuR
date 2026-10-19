import numpy as np
import scipy.linalg
from dynasaur.planning.core import Agent
from dynasaur.util.pink_noise import PinkNoiseGenerator
    
class LQR_Cartpole(Agent):
    def __init__(self, cfg):
        self.K = self.get_lqr_gain()
        self.pink_noise_exploration_generator = PinkNoiseGenerator(pink_noise_shape=(1,), pink_noise_length=cfg.algorithm.trial_length, device=cfg.device,seed = cfg.seed)
    def get_lqr_gain(self):
        m = 0.1
        M = 1.0
        L = 1.0
        g = 9.8
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -m * g / M, 0],
            [0, 0, 0, 1],
            [0, 0, (M + m) * g / (L * M), 0]
        ])

        B = np.array([[0], [1 / M], [0], [-1 / (L * M)]])

        Q = np.diag([10, 1, 10, 1])
        R = np.array([[0.1]])
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K  
    def act(
        self, obs: np.ndarray, sample: bool = False
    ) -> np.ndarray:
        noise = self.pink_noise_exploration_generator.get_pink_action().cpu().numpy() * 0.4
        controll = np.squeeze(-1*self.K@obs)
        if sample == True:
            controll = np.clip(controll+noise,-1,1)
        return np.clip(controll,-1,1)
    def resetAgent(self):
        self.pink_noise_exploration_generator.reset_pink()
