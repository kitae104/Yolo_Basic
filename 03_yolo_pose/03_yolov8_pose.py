import cv2
import time
from ultralytics import YOLO

print('Starting...')

model = YOLO('yolov8n-pose.pt')     # 모델 설정

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0
results = None

count = 0

while cap.isOpened():
  
  success, frame = cap.read()       # 프레임 읽기

  count = count + 1                 # 프레임 카운트
 
  if success:
      # 프레임 크기 조정                              
      # frame = cv2.resize(frame, (width, height))

      # 2프레임 마다 YOLOv8 추론 수행
      if time.time() - start_time >= 0.5:          
          results = model(frame)                # YOLOv8 추론 수행          
          start_time = time.time()              # 시간 초기화
          #frame_count += 2

      # 프레임에 결과 시각화      
      if results is not None:                   # 결과가 존재하면
          annotated_frame = results[0].plot()   # YOLOv8 결과 시각화
          
          cv2.imshow("YOLOv8 Inference", annotated_frame)   # 시각화된 프레임 출력

          P = 0
          try:
              # 사람이 검출되었는지 확인
              for idx, kpt in enumerate(results[0].keypoints[0]):
                  print('Persons Detected')
                  P = 1         # 사람이 검출되면 P = 1
          except:
                  print('No Persons')
                  P = 0         # 사람이 검출되지 않으면 P = 0
      
      if cv2.waitKey(1) & 0xFF == ord("q"):   # 'q' 키를 누르면 종료
          break
  else:      
      break                          # 프레임 읽기 실패시 종료
 
cap.release()
cv2.destroyAllWindows()
