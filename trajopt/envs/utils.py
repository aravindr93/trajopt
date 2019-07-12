from trajopt.envs.reacher_env import Reacher7DOFEnv
from trajopt.envs.continual_reacher_env import ContinualReacher7DOFEnv
from adept_envs.dclaw.pose import DClawPoseStill

import gym
import mjrl.envs
import adept_envs
import trajopt.envs
from mjrl.utils.gym_env import GymEnv

def get_environment(env_id):
    return GymEnv(env_id)

# def get_environment(env_id):
#     if env_id == 'reacher_7dof':
#         return Reacher7DOFEnv()
#     elif env_id == 'continual_reacher_7dof':
#         return ContinualReacher7DOFEnv()
#     elif env_id == 'DClawPoseStill-v0':
#         return DClawPoseStill()
#     else:
#         print("Unknown Environment Name")
#         return None
