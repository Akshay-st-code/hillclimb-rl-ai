# capture_and_display.py

import pyautogui
import time
import os
from datetime import datetime
import cv2
import numpy as np

# Folder to save screenshots
SAVE_DIR = "screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define screen region for capture (x, y, width, height)
region = (0, 0, 800, 480)  # ⚠️ Adjust this to your game screen

print("📸 Screenshot capture started. Press Ctrl+C to stop.")

try:
    count = 0
    while True:
        # Take screenshot
        screenshot = pyautogui.screenshot(region=region)

        # Convert to OpenCV format (numpy array)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Show image in a window
        cv2.imshow("Live Screenshot", frame)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}_{count}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

        # Delay
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

finally:
    cv2.destroyAllWindows()
