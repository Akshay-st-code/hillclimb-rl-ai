# evaluate.py

from env import HillClimbEnv
from dqn_agent import DQNAgent
import time

MODEL_PATH = "C:\\Users\\aksha\\Desktop\\AI_Hill_Climb_Racing\\runs\\detect\\train13\\weights\\best.pt"

env = HillClimbEnv(MODEL_PATH)
agent = DQNAgent(state_dim=5, action_dim=3)
agent.load("dqn_checkpoint.pth")  # Load best model

print("🎮 Running trained agent...")

while True:
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        print(f"Action: {action} | Reward: {reward:.2f}", end="\r")
        time.sleep(0.1)
