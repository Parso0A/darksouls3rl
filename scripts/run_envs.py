import fire
import gymnasium
import datetime
import logging

import gymnasium.wrappers.flatten_observation
import soulsgym  # noqa: F401
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from keras import __version__
tf.keras.__version__ = __version__
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

experience_buffer = []
def main(env: str = "SoulsGymIudex-v0"):
    soulsgym.set_log_level(logging.DEBUG)
    gymnasium.logger.set_level(40)
    env = gymnasium.make(env, game_speed=3.)
    env.reset()
    #action = env.action_space.sample()
    #observation, r, done,trunc, info= env.step(action)
    print(env.obs)
    states = len(env.obs)#gymnasium.wrappers.flatten_observation.FlattenObservation(env).observation_space.shape[0] #observation.shape[0]
    actions = env.action_space.n
    # model = Sequential()
    
    # model.add(Flatten(input_shape=(1, 2)))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(actions, activation='linear'))
    
    # agent = DQNAgent(
    #     model=model,
    #     memory=SequentialMemory(limit=50000, window_length=1),
    #     policy=BoltzmannQPolicy(),
    #     nb_actions=actions,
    #     nb_steps_warmup=100,
    #     target_model_update=0.01
    # )
    # agent.compile(Adam(lr=0.001), metrics=['mae'])
    # agent.fit(env, nb_steps=100000, visualize=False, verbose=1)
    
    # results = agent.test(env, nb_episodes=10, visualize=True)

    # print(np.mean(results.history['episode_reward']))

    try:
        start = datetime.datetime.now()
        while True:
            env.reset()
            terminated, truncated = False, False
            obs = env.obs
            while not terminated and not truncated:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print( reward, next_obs)
                store_experience(obs, action, reward, next_obs, done)
                obs = next_obs
            print("Current runtime: " + str(datetime.datetime.now() - start).split('.')[0])
    finally:
        with open('experience_buffer.pkl', 'wb') as f:
            pickle.dump(experience_buffer, f)
        env.close()
        #plot()

def store_experience(obs, action, reward, next_obs, done):
    if not isinstance(obs, dict) or not isinstance(next_obs, dict):
        raise ValueError("Observations must be dictionaries")
    if not isinstance(reward, float):
        raise ValueError("Action must be an integer and reward must be a float")
    if not isinstance(done, bool):
        raise ValueError("Done flag must be a boolean")
    experience_buffer.append((obs, action, reward, next_obs, done))

def plot():
    # Load the experience buffer from the file
    with open('experience_buffer.pkl', 'rb') as f:
        experience_buffer = pickle.load(f)

        # Preprocess the data
    # Extract the data from the experience buffer
    obs = [x[0] for x in experience_buffer]
    obs_values = []
    for x in obs:
        for k, v in x.items():
            if isinstance(v, (int, float)):
                obs_values.append(v)
    actions = [x[1] for x in experience_buffer]
    rewards = [x[2] for x in experience_buffer]
    done = [x[4] for x in experience_buffer]

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, figsize=(10, 10))

    # Plot observations in the first subplot
    axs[0].plot(obs_values)
    axs[0].set_title('Observations')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Value')

    # Plot actions in the second subplot
    axs[1].plot(actions)
    axs[1].set_title('Actions')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Value')

    # Plot rewards in the third subplot
    axs[2].plot(rewards)
    axs[2].set_title('Rewards')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Value')

    # Plot done flags in the fourth subplot
    axs[3].plot(done)
    axs[3].set_title('Done Flags')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Value')

    # Layout so plots do not overlap
    fig.tight_layout()

    plt.show()

# if __name__ == "__main__":
#     plot()

# def main(boss: str):
#     env = gymnasium.make(f"SoulsGym{boss}-v0")
#     try:
#         for _ in range(3):
#             obs, info = env.reset()
#             terminated, truncated = False, False
#             while not terminated and not truncated:
#                 next_obs, reward, terminated, truncated, info = env.step(19)
#     finally:
#         env.close()


if __name__ == "__main__":
    logging.basicConfig(filename="soulsgym.log",
                        filemode="w",
                        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    soulsgym.set_log_level(level=logging.DEBUG)
    fire.Fire(main)