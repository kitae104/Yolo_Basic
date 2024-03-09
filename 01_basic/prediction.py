from ultralytics import YOLO

model = YOLO("D:/Github/Vision_WS/Yolo_Basic/01_basic/yolov8n.pt")

# model.predict(source="D:/Github/Vision_WS/Yolo_Basic/01_basic/image1.jpg", save=True, conf=0.5, save_txt=True)

model.export(format="onnx")
