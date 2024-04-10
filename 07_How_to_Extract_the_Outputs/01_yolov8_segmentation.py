import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:

  def __init__(self, capture_index):
    self.capture_index = capture_index
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", self.device)
    self.model = self.load_model()

  def load_model(self):
    model = YOLO("yolov8m.pt")
    model.fuse()
    return model
  
  def predict(self, frame):
    results = self.model(frame)
    return results
  
  def plot_bboxes(self, frame, results):
    xyxys = []
    confiendces = []
    class_ids = []

    for result in results:
      boxes = result.boxes.cpu().numpy()   
      # xyxy = boxes.xyxy
      # print(xyxy)

      # for xyxy in xyxys:
      #   cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]), (0, 255, 0), 2))

      xyxys.apend(boxes.xyxy)
      confidences.append(boxes.conf)
      class_ids.append(boxes.cls)
      results[0]
    return frame



  

