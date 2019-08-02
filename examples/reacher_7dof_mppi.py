from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle

# =======================================
ENV_NAME = 'trajopt_reacher-v0'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 12345
N_ITER = 1
H_total = 75
# =======================================

e = get_environment(ENV_NAME)
e.reset(seed=SEED)
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, 0.25, 0.8, 0.0]

agent = MPPI(e, H=16, paths_per_cpu=25, num_cpu=1,
             kappa=5.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
             default_act='mean', seed=SEED)

ts = timer.time()
for t in tqdm(range(H_total)):
    agent.train_step(niter=N_ITER)
    if t % 75 == 0 and t > 0:
        print("==============>>>>>>>>>>> saving progress ")
        pickle.dump(agent, open(PICKLE_FILE, 'wb'))

pickle.dump(agent, open(PICKLE_FILE, 'wb'))
print("Trajectory reward = %f" % np.sum(agent.sol_reward))
print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

# wait for user prompt before visualizing optimized trajectories
_ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
for _ in range(100):
    agent.animate_result()
