# CONCI01-RL

The Reinforcement Learning environment created follows the OpenAI Gym interface, so it can be used with many Reinforcement Learning Frameworks including [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/), [OpenAI Baselines](https://github.com/openai/baselines), and [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/). The framework of choice was Stable Baselines3, and this user guide will explain the basic usage with Stable Baselines3. The full features of Stable Baselines3 can be found in the [SB3 Documentation](https://stable-baselines3.readthedocs.io/en/master/index.html) and [RL Baselines3 Zoo Github](https://github.com/DLR-RM/rl-baselines3-zoo)

**Not runnable! Lacks the confidential ML models and historical race data** 

## Installation 
- Install Stable Baselines3, following the [SB3 Installation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- Install RL Baselines3 Zoo, following the [RL Baselines3 Zoo Installation](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
- Clone the CONCI01-RL repository and set the directory to the cloned folder.
- Install the environments in CONCI01-RL using pip:
```pip install -e .```
- Add the following lines to *rl-baselines3-zoo/utils/import_envs.py*:
```
try:
    import gym_jtr
except ImportError:
    gym_jtr = None
```
 

## Train
- Switch to rl-baselines3-zoo folder.
- Hyperparameters can be specified from the files in hyperparams folder. If not, the default hyperparameters will be used.
- Train the agent with selected algorithm on the selected environment:
    ```python train.py --algo algo_name --env env_id```
For example:
    ```python train.py --algo sac --env jtr-modelless-v4```
- Details of training options can be found in RL Baselines3 Zoo [Docs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html) and [Github](https://github.com/DLR-RM/rl-baselines3-zoo)

## View Results
The resulting agents can be tested on the environment by:
```python enjoy.py --algo algo_name --env env_id --folder folder_name```
For example:
```python enjoy.py --algo sac --env jtr-modelless-v4 --folder logs/```

Or different graphs can be viewed by
- ```scripts/all\_plots.py``` or ```scripts/plot\_from\_file.py``` for plotting evaluations
- ```scripts/plot\_train.py``` for plotting training reward/success

The most used plotting option was scripts/plot\_train.py, such as:
```python scripts/plot_train.py -a sac -env jtr-modelless-v4 -f logs/```

Again, details of options can be found in RL Baselines3 Zoo [Docs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html) and [Github](https://github.com/DLR-RM/rl-baselines3-zoo)

It should be noted that the plotting options are buggy, especially on shorter training runs. They sometimes fail even on official OpenAI Gym environments.

## Trained Models
Trained models can be found in /logs folder, which contains another README file.
