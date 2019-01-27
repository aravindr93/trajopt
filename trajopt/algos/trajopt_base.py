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
        self.env.mujoco_render_frames = True
        for k in range(act.shape[0]):
            # self.env.mj_render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        self.env.mujoco_render_frames = False

    def animate_result(self):
        self.env.mujoco_render_frames = True
        for k in range(len(self.sol_act)):
            self.env.set_env_state(self.sol_state[k])
            self.env.step(self.sol_act[k])
        self.env.mujoco_render_frames = False

    def animate_result_offscreen(self, save_loc='/tmp/', filename='newvid', frame_size=(640,480), camera_id=0):
        import skvideo.io
        # self.env.mujoco_render_frames = True
        arrs = []
        for k in range(len(self.sol_act)):
            self.env.set_env_state(self.sol_state[k])
            self.env.step(self.sol_act[k])
            curr_frame = self.env.sim.render(width=frame_size[0], height=frame_size[1], camera_id=camera_id)
            arrs.append(curr_frame)
            print(k, end=', ', flush=True)

        file_name = save_loc + filename + ".mp4"
        skvideo.io.vwrite( file_name, np.asarray(arrs))
        print("saved", file_name)
        # self.mujoco_render_frames = False
