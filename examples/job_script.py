"""
Job script to optimize trajectories with trajopt
See: https://github.com/aravindr93/trajopt.git
"""

from trajopt.algos.mppi import MPPI
from trajopt.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
import gym
import mjrl.envs
import trajopt.envs
import mj_envs
import argparse
import json
import os
import dmc2gym

# =======================================
# Get command line arguments
parser = argparse.ArgumentParser(
    description="Trajectory Optimization with filtered MPPI"
)
parser.add_argument(
    "--output", type=str, required=True, help="location to store results"
)
parser.add_argument(
    "--config", type=str, required=True, help="path to job data with exp params"
)
args = parser.parse_args()
OUT_DIR = args.output
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with open(args.config, "r") as f:
    job_data = eval(f.read())

# Unpack args and make files for easy access
ENV_NAME = job_data["env_name"]
PICKLE_FILE = OUT_DIR + "/trajectories.pickle"
MJRL_PATH_FILE = OUT_DIR + "/mjrl_paths.pickle"
EXP_FILE = OUT_DIR + "/job_data.json"
SEED = job_data["seed"]
with open(EXP_FILE, "w") as f:
    json.dump(job_data, f, indent=4)
if "visualize" in job_data.keys():
    VIZ = job_data["visualize"]
else:
    VIZ = False
if "visualize_offscreen" in job_data.keys():
    VIZ_OS = job_data["visualize_offscreen"]
else:
    VIZ_OS = False

# helper function for visualization
def trigger_tqdm(inp, viz=False):
    if viz:
        return tqdm(inp)
    else:
        return inp


# =======================================
# Train loop
e = get_environment(ENV_NAME)
mean = np.zeros(e.action_dim)
sigma = 1.0 * np.ones(e.action_dim)
filter_coefs = [
    sigma,
    job_data["filter"]["beta_0"],
    job_data["filter"]["beta_1"],
    job_data["filter"]["beta_2"],
]
trajectories = []  # TrajOpt format (list of trajectory classes)
paths = []  # MJRL format (list of dictionaries)

ts = timer.time()
for i in range(job_data["num_traj"]):
    start_time = timer.time()
    print("Currently optimizing trajectory : %i" % i)
    seed = job_data["seed"] + i * 12345
    e.set_seed(seed)
    e.reset()
    e.reset()

    agent = MPPI(
        e,
        H=job_data["plan_horizon"],
        paths_per_cpu=job_data["paths_per_cpu"],
        num_cpu=job_data["num_cpu"],
        kappa=job_data["kappa"],
        gamma=job_data["gamma"],
        mean=mean,
        filter_coefs=filter_coefs,
        default_act=job_data["default_act"],
        seed=seed,
    )

    for t in tqdm(range(job_data["H_total"])):
        agent.train_step(job_data["num_iter"])
        if t % 20 == 0 and t > 0:
            SAVE_FILE = OUT_DIR + "/traj_%i.pickle" % i
            pickle.dump(agent, open(SAVE_FILE, "wb"))

    end_time = timer.time()
    print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    print("Optimization time for this trajectory = %f" % (end_time - start_time))
    trajectories.append(agent)
    pickle.dump(trajectories, open(PICKLE_FILE, "wb"))

    # make mjrl path object
    path = dict(
        observations=np.array(agent.sol_obs[:-1]),
        actions=np.array(agent.sol_act),
        rewards=np.array(agent.sol_reward),
        states=agent.sol_state,
    )
    paths.append(path)
    pickle.dump(paths, open(MJRL_PATH_FILE, "wb"))

print("Time for trajectory optimization = %f seconds" % (timer.time() - ts))
pickle.dump(trajectories, open(PICKLE_FILE, "wb"))

if VIZ_OS:
    import skvideo.io

    for idx, traj in enumerate(trajectories):
        VID_FILE = OUT_DIR + "/viz_" + str(idx) + ".mp4"
        frames = traj.animate_result_offscreen()
        skvideo.io.vwrite(VID_FILE, np.asarray(frames))
if VIZ:
    _ = input(
        "Press enter to display optimized trajectory (will be played 10 times) : "
    )
    for i in range(10):
        [traj.animate_result() for traj in trajectories]

# =======================================
