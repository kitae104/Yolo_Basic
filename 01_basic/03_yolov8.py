import cv2
from ultralytics import YOLO

# 모델 설정 
model = YOLO('yolov8n.pt')

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()             # 프레임 읽기

    if success:        
        results = model(frame)              # YOLOv8 추론 수행
        
        annotated_frame = results[0].plot() # YOLOv8 결과 시각화
        
        cv2.imshow("YOLOv8 Inference", annotated_frame) # 시각화된 프레임 출력
        
        if cv2.waitKey(1) & 0xFF == ord("q"):# 'q' 키를 누르면 종료
            break
    else:        
        break                   # 프레임 읽기 실패시 종료

cap.release()                   # 비디오 캡처 자원 반납
cv2.destroyAllWindows()         # 창 닫기