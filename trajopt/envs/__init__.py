from gym.envs.registration import register

# ----------------------------------------
# trajopt environments
# ----------------------------------------
# max_episode_steps is not used internally in trajopt
# value picked based on requirements of RL

register(
    id="trajopt_reacher-v0",
    entry_point="trajopt.envs:Reacher7DOFEnv",
    max_episode_steps=75,
)

register(
    id="trajopt_continual_reacher-v0",
    entry_point="trajopt.envs:ContinualReacher7DOFEnv",
    max_episode_steps=250,
)

from trajopt.envs.reacher_env import Reacher7DOFEnv, ContinualReacher7DOFEnv
