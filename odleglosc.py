from ultralytics import YOLO
import cv2
import time
model = YOLO("best.pt")
print(model.names)