from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
import time as timer
import numpy as np
import pickle

# =======================================
ENV_NAME = 'reacher_7dof'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 12345
H_total = 100
# =======================================

e = get_environment(ENV_NAME)
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, 0.25, 0.0, 0.0]

agent = MPPI(e, H=16, paths_per_cpu=8, num_cpu=6,
             kappa=25.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
             default_act='mean', seed=SEED)

ts = timer.time()
for t in range(H_total):
    print("Agent timestep : %i" % t)
    agent.train_step()
    if t % 25 == 0 and t > 0:
        print("==============>>>>>>>>>>> saving progress ")
        pickle.dump(agent, open(PICKLE_FILE, 'wb'))

pickle.dump(agent, open(PICKLE_FILE, 'wb'))
print("Trajectory reward = %f" % np.sum(agent.sol_reward))
print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

# wait for user prompt before visualizing optimized trajectories
_ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
for _ in range(100):
    agent.animate_result()