from ultralytics import YOLO
import cv2

# Load your best model
model = YOLO("runs/detect/train13/weights/best.pt")

# Load a test image (game screenshot)
img = cv2.imread("30.png")
results = model(img)

# Show results
cv2.imshow("Prediction", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
