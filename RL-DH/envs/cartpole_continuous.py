import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import torch
def double_to_hyperplane(action):
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

class CartPoleEnv(gym.Env):
    # This is a continuous version of gym's cartpole environment, with the only difference
    # being valid actions are any numbers in the range [-1, 1], and the are applied as
    # a multiplicative factor to the total force.
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None, screen_width: int = 600, screen_height: int =400):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 4.8

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.state = None

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_trunc = 0 
        self.steps_beyond_terminated = None

    def step(self, action):
        action = action.squeeze()
        x, x_dot, theta, theta_dot = self.state
        for i in range(1):
            force = action * self.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_dot**2 * sintheta
            ) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == "euler":
                    x = x + self.tau * x_dot
                    x_dot = x_dot + self.tau * xacc
                    theta = theta + self.tau * theta_dot
                    theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward =np.exp(-0.2* np.power((x-4.1),2))
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -100
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        self.steps_trunc +=1
        truncated = True if self.steps_trunc>=500 else False
        return np.array(self.state), reward, terminated, truncated, {"cost": float(terminated)*100}
    

    def step_batch(self, state_batch, action_batch):
        force_mag = 10.0
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02  # seconds between state updates
        kinematics_integrator = "euler"
        action_batch = action_batch.squeeze()
        x, x_dot, theta, theta_dot = state_batch.T
        force = action_batch * force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        if kinematics_integrator == "euler":
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
        next_state = np.stack([x, x_dot, theta, theta_dot], axis=1)
        terminated = np.logical_or(
            x < -self.x_threshold, 
            np.logical_or(
                x > self.x_threshold ,
                np.logical_or(
                theta < -self.theta_threshold_radians,
                theta > self.theta_threshold_radians
                )
            )
        )
        return next_state, terminated
    
    def step_batch_torch(self, state_batch, action_batch):
        
        force_mag = 10.0
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02  # seconds between state updates
        kinematics_integrator = "euler"
        action_batch = action_batch.squeeze()
        x, x_dot, theta, theta_dot = state_batch.T
        force = action_batch * force_mag
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (
            force + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        if kinematics_integrator == "euler":
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
        next_state = torch.stack([x, x_dot, theta, theta_dot], dim=1)
        terminated = torch.logical_or(
            x < -self.x_threshold, 
            torch.logical_or(
                x > self.x_threshold ,
                torch.logical_or(
                theta < -self.theta_threshold_radians,
                theta > self.theta_threshold_radians
                )
            )
        )
        return next_state, terminated
    
    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)
        if options == None:
            options = {}
        self.steps_trunc=0
        self.state = self.np_random.uniform(low=-options.get("start_deviation",0.05), high=options.get("start_deviation",0.05), size=(4,))
        #self.state[0] = np.random.uniform(low=-(self.x_threshold-0.3), high=self.x_threshold-0.3)
        #self.state[0] = np.random.uniform(low=-0.3, high=0.3)
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return np.array(self.state), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame  # type: ignore
            from pygame import gfxdraw  # type: ignore
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2 + 0.4  # 0.4 for extra space
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        gfxdraw.vline(self.surf,int(self.x_threshold * scale + self.screen_width / 2.0), 0, self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(-self.x_threshold * scale + self.screen_width / 2.0),0,self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(4.1 * scale + self.screen_width / 2.0),0,self.screen_height, (0, 255, 255))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def render_extra(self, action_hyperplane=0.0, action_proposal=0.0, filtered_action=0.0):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame  # type: ignore
            from pygame import gfxdraw  # type: ignore
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2 + 0.4  # 0.4 for extra space
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        gfxdraw.vline(self.surf,int(self.x_threshold * scale + self.screen_width / 2.0), 0, self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(-self.x_threshold * scale + self.screen_width / 2.0),0,self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(4.1 * scale + self.screen_width / 2.0),0,self.screen_height, (0, 255, 255))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        # Draw hyperplane overlay
        self._draw_hyperplane_overlay(self.screen, action_hyperplane, action_proposal, filtered_action)
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def _draw_hyperplane_overlay(self, screen, action_hyperplane, action_proposal, filtered_action):
        """Draw hyperplane visualization overlay on the CartPole screen"""
        try:
            import pygame
        except ImportError:
            return
        
        # Convert tensors to Python floats if needed
        if torch.is_tensor(action_hyperplane):
            action_hyperplane = action_hyperplane.item()
        if torch.is_tensor(action_proposal):
            action_proposal = action_proposal.item()
        filtered_action = filtered_action.item()
        direction_h, b_h = double_to_hyperplane(action_hyperplane)
        
        # Define colors as RGB tuples
        COLOR_SAFE = (0, 92, 0)  # #005c00
        COLOR_UNSAFE = (196, 32, 33)  # #c42021
        COLOR_PROPOSAL = (236, 159, 5)  # ec9f05
        
        # Create overlay surface with transparency
        overlay = pygame.Surface((200, 150))
        overlay.set_alpha(220)
        overlay.fill((240, 240, 240))
        
        # Draw border
        pygame.draw.rect(overlay, (0, 0, 0), overlay.get_rect(), 2)
        
        # Font for text
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Title
        title_text = font.render("Viable Action Interval", True, (0, 0, 0))
        overlay.blit(title_text, (10, 10))
        #title_text = font.render(f"Direction H: {direction_h:.3f}", True, (0, 0, 0))
        #overlay.blit(title_text, (10, 20))
        #title_text = font.render(f"Offset: {b_h:.3f}", True, (0, 0, 0))
        #overlay.blit(title_text, (10, 30))
        # Action values
        proposal_text = small_font.render(f"Proposal Action: {action_proposal:.2f}", True, COLOR_PROPOSAL)
        overlay.blit(proposal_text, (10, 35))
        
        # Check constraint satisfaction
        satisfies = direction_h * action_proposal < b_h
        if satisfies:
            filtered_text = small_font.render(f"Filtered Action: {filtered_action:.2f}", True, COLOR_SAFE)
            overlay.blit(filtered_text, (10, 50))
            status_text = small_font.render("Proposal action is unviable", True, COLOR_UNSAFE)
        else:
            filtered_text = small_font.render(f"Filtered Action: {action_proposal:.2f}", True, COLOR_SAFE)
            overlay.blit(filtered_text, (10, 50))
            status_text = small_font.render("Proposal action is viable", True, COLOR_SAFE)
        
        overlay.blit(status_text, (10, 70))
        
        # Draw action space visualization
        action_viz_y = 100
        action_viz_width = 180
        action_viz_height = 15
        
        # Action space background
        pygame.draw.rect(overlay, (200, 200, 200), 
                        (10, action_viz_y, action_viz_width, action_viz_height))
        
        # Feasible/infeasible regions
        if direction_h > 0:
            feasible_start = max(0, int(10 + (b_h+1)/2 * action_viz_width))
            pygame.draw.rect(overlay, COLOR_SAFE, 
                            (feasible_start, action_viz_y, 
                            10 + action_viz_width - feasible_start, action_viz_height))

            infeasible_end = min(action_viz_width, int((b_h + 1)/2 * action_viz_width))
            pygame.draw.rect(overlay, COLOR_UNSAFE, 
                            (10, action_viz_y, infeasible_end, action_viz_height))
        else:
            feasible_end = min(action_viz_width, int((-1*b_h + 1)/2 * action_viz_width))
            pygame.draw.rect(overlay, COLOR_SAFE, 
                            (10, action_viz_y, feasible_end, action_viz_height))
            infeasible_start = max(0, int((-1*b_h + 1)/2 * action_viz_width))
            pygame.draw.rect(overlay, COLOR_UNSAFE, 
                            (10 + infeasible_start, action_viz_y, 
                            action_viz_width - infeasible_start, action_viz_height))
        
        # Proposal action marker
        proposal_x = int(10 + (action_proposal + 1) * action_viz_width / 2)
        pygame.draw.circle(overlay, COLOR_PROPOSAL, 
                          (proposal_x, action_viz_y + action_viz_height // 2), 4)
        
        # If proposal action is NOT viable (satisfies is True), show projection line and filtered action
        if satisfies:
            filtered_x = int(10 + (filtered_action + 1) * action_viz_width / 2)
            # Draw dotted line from proposal to filtered action
            # Create dotted line effect by drawing small segments
            num_segments = 10
            for i in range(num_segments):
                if i % 2 == 0:  # Only draw every other segment for dotted effect
                    start_x = proposal_x + (filtered_x - proposal_x) * i / num_segments
                    end_x = proposal_x + (filtered_x - proposal_x) * (i + 1) / num_segments
                    pygame.draw.line(overlay, (0, 0, 0), 
                                   (start_x, action_viz_y + action_viz_height // 2),
                                   (end_x, action_viz_y + action_viz_height // 2), 2)
            
            # Draw filtered action as black dot
            pygame.draw.circle(overlay, (0, 0, 0), 
                             (filtered_x, action_viz_y + action_viz_height // 2), 4)
        constraint_line = b_h / direction_h
        constraint_x = int(10 + (constraint_line + 1)/2 * action_viz_width)
        pygame.draw.line(overlay, (0, 0, 0), 
                        (constraint_x, action_viz_y-4), 
                        (constraint_x, action_viz_y + action_viz_height+4), 3)
        # Action space labels
        label_font = pygame.font.Font(None, 16)
        left_label = label_font.render("-1", True, (0, 0, 0))
        right_label = label_font.render("1", True, (0, 0, 0))
        middle_label = label_font.render("0", True, (0, 0, 0))
        overlay.blit(left_label, (5, action_viz_y + action_viz_height + 2))
        overlay.blit(right_label, (action_viz_width + 5, action_viz_y + action_viz_height + 2))
        overlay.blit(middle_label, (action_viz_width // 2 + 5, action_viz_y + action_viz_height + 2))
        
        # Blit overlay to main screen
        screen.blit(overlay, (self.screen_width - 220, 20))

    def close(self):
        if self.screen is not None:
            import pygame  # type: ignore

            pygame.display.quit()
            pygame.quit()
            self.isopen = False