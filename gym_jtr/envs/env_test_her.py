from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from jtr_env_her import JtrEnvHER

model_class = SAC  # works also with SAC, DDPG and TD3

env = JtrEnvHER()

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = 480

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)

# Train the model
model.learn(100)

model.save("./her_jtr_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load('./her_jtr_env', env=env)

obs = env.reset()
print('reset')
print('obs: {}'.format(obs))
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    print('predict')
    print('action: {}'.format(action))
    obs, reward, done, info = env.step(action)
    print('step')
    print('reward: {}'.format(reward))
    print('done: {}'.format(done))
    print('info: {}'.format(info))
    env.render()
    if done:
        print('done')
        obs = env.reset()
        print(obs)
obs = env.reset()
print(obs)