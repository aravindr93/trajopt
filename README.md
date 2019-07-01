# Trajectory Optimization Algorithms
This package contains trajectory optimization algorithms that are predominantly meant for continuous control taks (simulated with [MuJoCo](http://www.mujoco.org/)).

# Installation
The main package dependencies are `python>=3.5`, `gym`, `mujoco_py`, and `numpy`. The algorithms assume an environment abstraction similar to [`mj_envs`](https://github.com/vikashplus/mj_envs), which builds on top of the `gym` abstraction.

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

# Example Usage
See this directory for illustrative examples: [`trajopt/sandbox/examples`](https://github.com/aravindr93/trajopt/tree/master/trajopt/sandbox/examples).

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

