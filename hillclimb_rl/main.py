
# main.py

from env import HillClimbEnv
from dqn_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

MODEL_PATH = "C:\\Users\\aksha\\Desktop\\AI_Hill_Climb_Racing\\runs\\detect\\train13\\weights\\best.pt"
EPISODES = 500
MAX_STEPS = 300
CHECKPOINT_EVERY = 20
TARGET_UPDATE = 10

env = HillClimbEnv(MODEL_PATH)
agent = DQNAgent(state_dim=5, action_dim=3)
writer = SummaryWriter()

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        total_reward += reward
        state = next_state
        if done: break

    writer.add_scalar("Total Reward", total_reward, episode)

    if episode % TARGET_UPDATE == 0:
        agent.update_target()
    if episode % CHECKPOINT_EVERY == 0:
        agent.save()

    print(f"[Episode {episode+1}] Total Reward: {total_reward:.2f}")
print("⚙️ Warming up replay buffer...")
state = env.reset()
for _ in range(100):
    action = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])  # gas-biased
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()
print(f"[Ep {episode}] Step {step}: Action={action} | Reward={reward:.2f} | Car X={state[0]:.1f}")
