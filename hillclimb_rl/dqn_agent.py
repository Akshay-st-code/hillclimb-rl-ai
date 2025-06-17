# dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=20000)


    
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target()
        self.steps = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 10000

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        epsilon = max(self.epsilon_end, self.epsilon_start * np.exp(-1. * self.steps / self.epsilon_decay))

        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps / self.epsilon_decay)
        self.steps += 1

        if random.random() < epsilon:
            # Gas-biased exploration
            weights = [0.1, 0.8, 0.1]  # none, gas, brake
            action = np.random.choice([0, 1, 2], p=weights)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        next_q = self.target_net(next_states_tensor).max(1)[0].detach().unsqueeze(1)
        target_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="dqn_checkpoint.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="dqn_checkpoint.pth"):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

