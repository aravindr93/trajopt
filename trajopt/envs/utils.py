from trajopt.envs.reacher_env import Reacher7DOFEnv
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from adept_envs.dclaw.pose import DClawPoseStill

import gym
import mjrl.envs
import trajopt.envs
from mjrl.utils.gym_env import GymEnv

def get_environment(env_id):
    return GymEnv(env_id)

