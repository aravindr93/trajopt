from trajopt.algos.mppi import MPPI
from trajopt.utils import get_environment, set_seed
from tqdm import tqdm
import numpy as np
import hydra, time, pickle, argparse, json, os, multiprocessing
import gym, trajopt.envs
from omegaconf import DictConfig, OmegaConf

os.environ["MUJOCO_GL"] = "egl"
cwd = os.getcwd()


@hydra.main(config_path="configs", config_name="test", version_base="1.1")
def configure_jobs(config: dict) -> None:
    """Wrapper function to launch jobs with hydra."""

    print("========================================")
    print("Job Configuration")
    print("========================================")

    config = OmegaConf.structured(OmegaConf.to_yaml(config))
    config["cwd"] = cwd
    with open("job_config.json", "w") as fp:
        OmegaConf.save(config=config, f=fp.name)
    print(OmegaConf.to_yaml(config))
    # run the base loop
    trajopt_loop(config)


def trajopt_loop(config: dict) -> None:

    if os.path.isdir(config["job_name"]) == False:
        os.mkdir(config["job_name"])
    previous_dir = os.getcwd()
    os.chdir(config["job_name"])  # important! we are now in the directory to save data

    e = get_environment(config["env_name"])
    mean = np.zeros(e.action_dim)
    sigma = 1.0 * np.ones(e.action_dim)
    # import ipdb; ipdb.set_trace()
    filter_coefs = [
        sigma,
        config["filter"]["beta_0"],
        config["filter"]["beta_1"],
        config["filter"]["beta_2"],
    ]
    trajectories = []  # TrajOpt format (list of trajectory classes)
    paths = []  # MJRL format (list of dictionaries)

    ts = time.time()
    for i in range(config["num_traj"]):
        start_time = time.time()
        print("Currently optimizing trajectory : %i" % i)
        seed = config["seed"] + i * 12345

        # set the seed
        set_seed(seed)
        e.set_seed(seed)
        e.reset()

        # make the MPC agent
        agent = MPPI(
            env=e,
            H=config["plan_horizon"],
            paths_per_cpu=config["paths_per_cpu"],
            num_cpu=config["num_cpu"],
            kappa=config["kappa"],
            gamma=config["gamma"],
            mean=mean,
            filter_coefs=filter_coefs,
            default_act=config["default_act"],
            seed=seed,
        )

        for t in tqdm(range(config["traj_length"])):
            agent.train_step(config["num_iter"])
            if t % 20 == 0 and t > 0:
                SAVE_FILE = "traj_%i.pickle" % i
                pickle.dump(agent, open(SAVE_FILE, "wb"))

        end_time = time.time()
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Optimization time for this trajectory = %f" % (end_time - start_time))
        trajectories.append(agent)
        pickle.dump(trajectories, open("trajectories.pickle", "wb"))

        # make mjrl path object
        path = dict(
            observations=np.array(agent.sol_obs[:-1]),
            actions=np.array(agent.sol_act),
            rewards=np.array(agent.sol_reward),
            states=agent.sol_state,
        )
        paths.append(path)
        pickle.dump(paths, open("paths.pickle", "wb"))

    print("Time for trajectory optimization = %f seconds" % (time.time() - ts))
    pickle.dump(trajectories, open("trajectories.pickle", "wb"))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    configure_jobs()
