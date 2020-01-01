# Trajectory Optimization Algorithms
This package contains trajectory optimization algorithms meant predominantly for continuous control taks (simulated with [MuJoCo](http://www.mujoco.org/)).

# Installation
The main package dependencies are `MuJoCo` and `mjrl`. See [setup-instructions](https://github.com/aravindr93/mjrl/tree/master/setup) to get a working conda environment and setup dependencies.

After [`mujoco_py`](https://github.com/openai/mujoco-py) has been installed, the package can be used by either adding to path as:
```
export PYTHONPATH=<path/to/trajopt>$PYTHONPATH
```
or through the pip install module
```
$ cd trajopt
$ pip install -e .
```
The tricky part of the installation is likely to be `mujoco_py`. Please see [instructions and known issues](https://github.com/aravindr93/mjrl/tree/master/setup) for help.

# API and example usage
The algorithms assume an environment abstraction similar to OpenAI `gym`, but requires two additional functions to be able to run the algorithms provided here.
- `get_env_state()` should return a dictionary with all the information required to reconstruct the scene and dynamics. For most use cases, this can just be the `qpos` and `qvel`. However, in some cases, additional information may be required to construct scene and dynamics. For example, in multi-goal RL, we can represent virtual goals using sites as opposed to real joints.
- `set_env_state(state_dict)` should take in a dictionary, and use the contents of the dictionary to recreate the scene specified by the dictionary.
The example [reacher environment](https://github.com/aravindr93/trajopt/blob/redesign/trajopt/envs/reacher_env.py) has an illustrative.

# Example Usage
See this directory for illustrative examples: [`trajopt/examples`](https://github.com/aravindr93/trajopt/tree/master/examples).

# Bibliography
If you find the package useful, please cite the following paper.
```
@INPROCEEDINGS{Lowrey-ICLR-19,
    AUTHOR    = {Kendall Lowrey AND Aravind Rajeswaran AND Sham Kakade AND 
                 Emanuel Todorov AND Igor Mordatch},
    TITLE     = "{Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control}",
    BOOKTITLE = {ICLR},
    YEAR      = {2019},
}
```
