# DQN算法 强化学习案例03
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

def compute_loss(model, target, states, actions, rewards, next_states, dones, gamma):
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    loss = F.mse_loss(q_values, expected_q_values.detach())
    return loss

def update_target(model, target):
    target.load_state_dict(model.state_dict())

def train(env, model, target, optimizer, replay_buffer, gamma, batch_size, target_update):
    state = env.reset()
    episode_reward = 0
    while True:
        epsilon = 0.05
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model(torch.FloatTensor(state)).argmax().item()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            loss = compute_loss(model, target, torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones), gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if target_update > 0 and len(replay_buffer) % target_update == 0:
                update_target(model, target)
        if done:
            break
    return episode_reward

def main():
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    target = DQN(state_dim, action_dim)
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(10000)
    gamma = 0.99
    batch_size = 32
    target_update = 1000
    num_episodes = 1000
    for i in range(num_episodes):
        episode_reward = train(env, model, target, optimizer, replay_buffer, gamma, batch_size, target_update)
        print("Episode {}: {}".format(i+1, episode_reward))
    env.close()

if __name__ == '__main__':
    main()
