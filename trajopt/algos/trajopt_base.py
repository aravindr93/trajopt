"""
Base trajectory class
"""

import numpy as np

class Trajectory:
    def __init__(self, env, H=32, seed=123):
        self.env, self.seed = env, seed
        self.n, self.m, self.H = env.observation_dim, env.action_dim, H

        # following need to be populated by the trajectory optimization algorithm
        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.zeros((self.H, self.m))

    def update(self, paths):
        """
        This function should accept a set of trajectories
        and must update the solution trajectory
        """
        raise NotImplementedError

    def animate_rollout(self, t, act):
        """
        This function starts from time t in the solution trajectory
        and animates a given action sequence
        """
        self.env.set_env_state(self.sol_state[t])
        for k in range(act.shape[0]):
            try:
                self.env.env.env.mujoco_render_frames = True
            except AttributeError:
                self.env.render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        try:
            self.env.env.env.mujoco_render_frames = False
        except:
            pass

    def animate_result(self):
        self.env.reset(self.seed)
        self.env.set_env_state(self.sol_state[0])
        for k in range(len(self.sol_act)):
            self.env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step(self.sol_act[k])
        self.env.env.env.mujoco_render_frames = False
