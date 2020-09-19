"""
Job script to optimize trajectories with trajopt
See: https://github.com/aravindr93/trajopt.git
"""

from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
import gym
import mjrl.envs
import trajopt.envs
#import mj_envs
import argparse
import json
import os

# =======================================
# Get command line arguments
parser = argparse.ArgumentParser(description='Trajectory Optimization with filtered MPPI')
parser.add_argument('-o', '--output', type=str, required=True, help='location to store results')
parser.add_argument('-c', '--config', type=str, required=True, help='path to job data with exp params')
parser.add_argument('-i', '--include', type=str, required=False, default=None, help='package to include')
args = parser.parse_args()
OUT_DIR = args.output
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
if args.include:
    exec("import "+args.include)
# Unpack args and make files for easy access
ENV_NAME = job_data['env_name']
PICKLE_FILE = OUT_DIR + '/trajectories.pickle'
EXP_FILE = OUT_DIR + '/job_data.json'
SEED = job_data['seed']
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)
if 'visualize' in job_data.keys():
    VIZ = job_data['visualize']
else:
    VIZ =False

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
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
trajectories = []

ts=timer.time()
for i in range(job_data['num_traj']):
    start_time = timer.time()
    print("Currently optimizing trajectory : %i" % i)
    seed = job_data['seed'] + i*12345
    e.reset(seed=seed)
    
    agent = MPPI(e,
                 H=job_data['plan_horizon'],
                 paths_per_cpu=job_data['paths_per_cpu'],
                 num_cpu=job_data['num_cpu'],
                 kappa=job_data['kappa'],
                 gamma=job_data['gamma'],
                 mean=mean,
                 filter_coefs=filter_coefs,
                 default_act=job_data['default_act'],
                 seed=seed)
    
    for t in trigger_tqdm(range(job_data['H_total']), VIZ):
        agent.train_step(job_data['num_iter'])
    
    end_time = timer.time()
    print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    print("Optimization time for this trajectory = %f" % (end_time - start_time))
    trajectories.append(agent)
    pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
    
print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))
pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))

if VIZ:
    _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    for i in range(10):
        [traj.animate_result() for traj in trajectories]

# =======================================
