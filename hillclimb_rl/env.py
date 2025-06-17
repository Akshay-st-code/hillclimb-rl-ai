# env.py

import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

class HillClimbEnv:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.last_car_x = None
        self.restart_coords = (300, 300)  # Update this to your restart button position
        self.last_action = None
        self.gas_hold_frames = 0
        self.last_state = None

    def _get_screenshot(self):
        img = pyautogui.screenshot(region=(0, 0, 800, 480))  # Adjust to your game window
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def _parse_detections(self, results):
        state = np.zeros(5)
        car_x = None

        for box in results[0].boxes:
            cls = int(box.cls)
            x_center = float(box.xywh[0][0])
            if cls == 1:  # car
                car_x = x_center
                state[0] = x_center
            elif cls == 0: state[1] = 1  # fuel_level
            elif cls == 4: state[2] = 1  # fuel_can
            elif cls == 2: state[3] = 1  # coin
            elif cls == 3: state[4] = 1  # diamond

        return state, car_x

    def _calculate_reward(self, state, car_x):
        # If we can't calculate distance, set reward to 0 or -10 if car is gone
        if car_x is None or self.last_car_x is None:
            self.last_car_x = car_x
            return -10 if car_x is None else 0

        # Base reward on distance moved forward
        reward = car_x - self.last_car_x
        self.last_car_x = car_x

        # Reward shaping
        if state[2]: reward += 10  # fuel_can
        if state[3]: reward += 3   # coin
        if state[4]: reward += 5   # diamond
        if reward < 0.05: reward -= 2  # punish small movement

        # Encourage gas
        if self.last_action == 1:
            reward += 0.5

        return reward



    def _press_key(self, action, state):
        HOLD_FRAMES = 20  # 2 seconds if time.sleep(0.1)

        if not hasattr(self, 'gas_hold_frames'):
            self.gas_hold_frames = 0

        # Logic for holding gas
        if self.last_action == 1:  # still holding gas
            if action == 1:
                self.gas_hold_frames += 1
            else:
                # If gas has been held < HOLD_FRAMES, ignore any change
                if self.gas_hold_frames < HOLD_FRAMES:
                    action = 1
                    self.gas_hold_frames += 1
                else:
                    self.gas_hold_frames = 0
        elif action == 1:
            self.gas_hold_frames = 1  # new gas press

        # If action changed, update keys
        if self.last_action != action:
            pyautogui.keyUp("right")
            pyautogui.keyUp("left")

            if action == 1:
                pyautogui.keyDown("right")
                print("🟢 Gas pressed")
            elif action == 2:
                pyautogui.keyDown("left")
                print("🔴 Brake pressed")
            else:
                print("⚪ No action")

        self.last_action = action


    def _auto_restart_game(self):
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")
        pyautogui.click(self.restart_coords)
        time.sleep(2)

    def reset(self):
        self.last_car_x = None
        self.last_action = None
        self.gas_hold_frames = 0
        self._auto_restart_game()
        time.sleep(1)
        return np.zeros(5)

    def step(self, action):
        self._press_key(action, self.last_state if self.last_state is not None else np.zeros(5))
        time.sleep(0.1)
        frame = self._get_screenshot()
        results = self.model(frame)
        state, car_x = self._parse_detections(results)
        reward = self._calculate_reward(state, car_x)
        done = reward == -10

        if done:
            pyautogui.keyUp("right")
            pyautogui.keyUp("left")
            self.last_action = None

        self.last_state = state
        return state, reward, done, results
