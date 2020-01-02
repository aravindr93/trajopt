import gym
import mjrl.envs
import trajopt.envs
from mjrl.utils.gym_env import GymEnv

def get_environment(env_id):
    return GymEnv(env_id)

