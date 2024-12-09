import math
import random
import fire
import gymnasium
import datetime
import logging
import torch
import torch.nn as nn
import gymnasium.wrappers.flatten_observation
from dqn import DQN
#from scripts.replay_memory import ReplayMemory
import soulsgym  # noqa: F401
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch import optim
from replay_memory import Transition, ReplayMemory

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


steps_done = 0


experience_buffer = []
def main(env: str = "SoulsGymIudex-v0"):
    soulsgym.set_log_level(logging.DEBUG)
    gymnasium.logger.set_level(40)
    env = gymnasium.make(env, game_speed=3.)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, info = env.reset()

    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = gymnasium.wrappers.FlattenObservation(env)

    n_observations = env.observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    try:
        load_model(policy_net, target_net, 'model_checkpoint.pth')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        

    episode_durations = []

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        #print(batch.reward)
        reward_batch = torch.cat([torch.tensor([r], device=device, dtype=torch.float32) for r in batch.reward])
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    try:
        start = datetime.datetime.now()
        while True:
            state, info = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = select_action(state)  #env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                print( reward, obs)
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                #store_experience(obs, action, reward, next_obs, done)
                memory.push(state, action, next_state, reward)
                
                state = next_state
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)
            print("Current runtime: " + str(datetime.datetime.now() - start).split('.')[0])
    finally:
        save_model(policy_net, target_net, 'model_checkpoint.pth')
        env.close()
        #plot()

def save_model(policy_net, target_net, filename):
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
    }, filename)

def load_model(policy_net, target_net, filename):
    checkpoint = torch.load(filename)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])

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