import numpy as np, torch, os
import multiprocessing as mp
import concurrent.futures
from trajopt import tensor_utils, math_utils
from trajopt.gym_env import GymEnv
from tqdm import tqdm
from typing import Union


def get_environment(
    env: Union[str, GymEnv, callable], env_kwargs: dict = None
) -> GymEnv:
    """Construct the GymEnv class that wraps around low-level environments.

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        Environment in the GymEnv format.
    """
    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError
    return env


def do_env_rollout(
    env: Union[str, GymEnv, callable],
    start_state: dict,
    act_list: list,
    env_kwargs: dict = None,
) -> list:
    """Rollout action sequence in env from provided initial states.

    Instantiates requested environment. Sets environment to provided initial states.
    Then rollouts out provided action sequence. Returns result in paths format (list of dicts).

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        act_list:
            List of numpy arrays containing actions that will be rolled out in open loop.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """

    # Not all low-level envs are picklable. For generalizable behavior,
    # we have the option to instantiate the environment within the rollout function.
    e = get_environment(env, env_kwargs)
    e.reset()
    e.real_env_step(
        False
    )  # indicates simulation for purpose of trajectory optimization
    paths = []
    H = act_list[0].shape[0]  # horizon
    N = len(act_list)  # number of rollout trajectories (per process)
    for i in range(N):
        e.set_env_state(start_state)
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        ifo = e.get_env_infos()
        for k in range(H):
            obs.append(e.get_obs())
            act.append(act_list[i][k])
            env_info = ifo if e.get_env_infos() == {} else e.get_env_infos()
            env_infos.append(env_info)
            states.append(e.get_env_state())
            s, r, d, ifo = e.step(act[-1])
            rewards.append(r)

        path = dict(
            observations=np.array(obs),
            actions=np.array(act),
            rewards=np.array(rewards),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            states=states,
        )
        paths.append(path)

    return paths


def generate_perturbed_actions(base_act: np.ndarray, filter_coefs: list) -> np.ndarray:
    """Generate perturbed actions around a base action sequence"""
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0 * eps[i] + beta_1 * eps[i - 1] + beta_2 * eps[i - 2]
    return base_act + eps


def generate_paths(
    env: Union[str, GymEnv, callable],
    start_state: dict,
    base_act: np.ndarray,
    filter_coefs: list,
    base_seed: int,
    num_paths: int,
    env_kwargs: Union[dict, None] = None,
    *args,
    **kwargs,
) -> list:
    """Generates perturbed action sequences and then performs rollouts.

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        base_act:
            A numpy array of base actions to which we add noise to generate action sequences for rollouts.
        filter_coefs:
            We use these coefficients to generate colored for action perturbation
        base_seed:
            Seed for generating random actions and rollouts.
        num_paths:
            Number of paths to rollout.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """
    set_seed(base_seed)
    act_list = []
    for i in range(num_paths):
        act = generate_perturbed_actions(base_act, filter_coefs)
        act_list.append(act)
    paths = do_env_rollout(env, start_state, act_list, env_kwargs)
    return paths


def gather_paths_parallel(
    env: Union[str, GymEnv, callable],
    start_state: dict,
    base_act: np.ndarray,
    filter_coefs: list,
    base_seed: int,
    paths_per_cpu: int,
    env_kwargs: Union[dict, None] = None,
    num_cpu: int = 1,
    *args,
    **kwargs,
):
    """Parallel wrapper around the gather paths function."""

    if num_cpu == 1:
        input_dict = dict(
            env=env,
            start_state=start_state,
            base_act=base_act,
            filter_coefs=filter_coefs,
            base_seed=base_seed,
            num_paths=paths_per_cpu,
            env_kwargs=env_kwargs,
        )
        return generate_paths(**input_dict)

    # do multiprocessing only if necessary
    input_dict_list = []
    for i in range(num_cpu):
        cpu_seed = base_seed + i * paths_per_cpu
        input_dict = dict(
            env=env,
            start_state=start_state,
            base_act=base_act,
            filter_coefs=filter_coefs,
            base_seed=cpu_seed,
            num_paths=paths_per_cpu,
            env_kwargs=env_kwargs,
        )
        input_dict_list.append(input_dict)

    results = _try_multiprocess_mp(
        func=generate_paths,
        input_dict_list=input_dict_list,
        num_cpu=num_cpu,
        max_process_time=300,
        max_timeouts=4,
    )
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess_mp(
    func: callable,
    input_dict_list: list,
    num_cpu: int = 1,
    max_process_time: int = 500,
    max_timeouts: int = 4,
    *args,
    **kwargs,
):
    """Run multiple copies of provided function in parallel using multiprocessing."""

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [
        pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list
    ]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess_mp(
            func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1
        )

    pool.close()
    pool.terminate()
    pool.join()
    return results


def _try_multiprocess_cf(
    func: callable,
    input_dict_list: list,
    num_cpu: int = 1,
    max_process_time: int = 500,
    max_timeouts: int = 4,
    *args,
    **kwargs,
):
    """Run multiple copies of provided function in parallel using concurrent futures."""

    results = None
    if max_timeouts != 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
            submit_futures = [
                executor.submit(func, **input_dict) for input_dict in input_dict_list
            ]
            try:
                results = [f.result() for f in submit_futures]
            except TimeoutError as e:
                print(str(e))
                print("Timeout Error raised...")
            except concurrent.futures.CancelledError as e:
                print(str(e))
                print("Future Cancelled Error raised...")
            except Exception as e:
                print(str(e))
                print("Error raised...")
                raise e
    return results


def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def trigger_tqdm(inp, viz: bool = False):
    """Helper function for visualization."""
    if viz:
        return tqdm(inp)
    else:
        return inp
