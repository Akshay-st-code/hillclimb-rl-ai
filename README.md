# 🧠 Autonomous Hill Climb Racing AI using YOLOv8 + DQN

This project builds an AI agent that autonomously plays the 2D game *Hill Climb Racing* using reinforcement learning and computer vision. The agent perceives the game screen via screenshots, detects objects using YOLOv8, and interacts using keyboard simulation with PyAutoGUI.

---

## 🚀 Features

- 🎯 Real-time object detection using **YOLOv8**
- 🤖 Reinforcement Learning agent based on **Deep Q-Network (DQN)**
- ⌨️ Game control using **PyAutoGUI** (no internal APIs)
- 🔄 Auto restart and crash recovery
- 📊 TensorBoard logging for training progress
- 💾 Model saving & evaluation script for demonstrations

---

## 🗂️ Project Structure

```
hillclimb_rl/
├── env.py           # Custom RL environment
├── dqn_agent.py     # Deep Q-Network logic
├── main.py          # Training script
├── evaluate.py      # Run the trained agent
├── utils.py         # Helper functions (screenshot, detection, etc.)
├── model/           # Saved model checkpoints
├── runs/            # Logs for training/evaluation
├── datasets/        # YOLO training data
└── README.md        # This file
```

---

## 🧪 Setup Instructions

### ✅ 1. Clone the repository
```bash
git clone https://github.com/Akshay-st-code/hillclimb-rl-ai.git
cd hillclimb-rl-ai
```

### ✅ 2. Install dependencies
```bash
pip install -r requirements.txt
```

### ✅ 3. Download YOLOv8 weights (or use your trained model)
Place your trained YOLOv8 model at:
```
runs/detect/train/weights/best.pt
```

### ✅ 4. Run Training
```bash
python main.py
```

### ✅ 5. Run Evaluation (demo play)
```bash
python evaluate.py
```

---

## 🎮 Game Requirements

- Install Hill Climb Racing on Windows
- Set game window resolution to 800x480
- Place game window at top-left (0,0)
- Ensure it’s **active and focused** during play

---

## 📊 Results

| Metric         | Value     |
|----------------|-----------|
| YOLO mAP@0.5   | 95.3%     |
| Avg. Reward    | +10.5     |
| Max Distance   | 1134 px   |
| Fuel Pickups   | 6–7 per run |
| Crash Rate     | ↓ to 19% |

---

## 📚 Future Improvements

- PPO or A3C policy optimization
- GUI overlay with reward/live Q-values
- Transfer to other games (e.g., Mario, Dino)
- Model distillation for mobile use

---

## 👨‍💻 Author

**Podupuganti Akshay**  
B.Tech, Software Engineering – 2024-25  
Contact: [akshay25112003@gmail.com]

---

## 📄 License

This project is open-source for educational and research purposes. Commercial use prohibited without permission.
