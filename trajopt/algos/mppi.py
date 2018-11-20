"""
Basic Model Predictive Path Integral
"""

import numpy as np
from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils import gather_paths_parallel

class MPPI(Trajectory):
    def __init__(self, env, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 seed=123,
                 ):
        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean

    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)

        # accept first action and step
        self.sol_act.append(act_sequence[0])
        state_now = self.sol_state[-1].copy()
        self.env.set_env_state(state_now)
        _, r, _, _ = self.env.step(act_sequence[0])
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.sol_reward.append(r)

        # get updated action sequence
        self.act_sequence[:-1] = act_sequence[1:]
        if self.default_act == 'repeat':
            self.act_sequence[-1] = self.act_sequence[-2]
        else:
            self.act_sequence[-1] = self.mean.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["rewards"].shape[0]):
                scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
        return scores

    def do_rollouts(self, seed):
        paths = gather_paths_parallel(self.env.env_name,
                                      self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      )
        return paths

    def train_step(self):
        t = len(self.sol_state) - 1
        paths = self.do_rollouts(self.seed+t)
        self.update(paths)