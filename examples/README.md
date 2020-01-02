# Trajopt examples

This directory contains examples for running MPC algorithms on gym-like environments.

1. Clone and checkout the below repos by following setup instructions in individual repos:
    - [mjrl](https://github.com/aravindr93/mjrl) -- has detailed setup instructions [here](https://github.com/aravindr93/mjrl/tree/master/setup)
    - [trajopt](https://github.com/aravindr93/trajopt) -- this repository containing the MPC algorithms

2. Run a toy experiment (point mass) using:
    ```
    python job_script.py --output point_mass_job --config configs/point_mass_config.txt
    ```
    This optimization should take only a second on so. After the optimization is complete, there will be a prompt to visualize the optimized trajectory. This can be turned off by changing the debug field in the config file.

3. Make a 7DOF sawyer arm reach various goals with the end effector (either fixed goal in episodic setting, or changing goals in continual setting)
    ```
    python job_script.py --output reacher_7dof_job --config configs/reacher_7dof_config.txt
    python job_script.py --output continual_reacher_7dof_job --config configs/continual_reacher_7dof_config.txt
    ```

4. In-hand manipulation of a pen with a Shadow Hand (Adroit). 
    This requires installation of the [mj_envs](https://github.com/vikashplus/mj_envs) repository. This package uses git submodules, additional care must be taken to exactly follow setup instructions. **NOTE:** Uncomment the `import mj_envs` line in the `job_script.py` file.
    ```
    python job_script.py --output pen_job --config configs/pen_config.txt
    ```

5. To visualize previous runs (stored as pickle files)
    ```
    python visualize_trajectories.py --file <path/to/file.pickle> --repeat <#times to repeat>
    ```
