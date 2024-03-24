# 라이브러리 임포트
import torch

# YOLOv8 라이브러리에서 필요한 모듈 임포트
from models import YOLOv8  # YOLOv8 모델 임포트
from utils.datasets import LoadImagesAndLabels  # 데이터셋 로딩
from utils.train import train_detector  # 학습 함수

# 학습 파라미터 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU 사용 가능 여부 확인
data_path = '/content/drive/MyDrive/FootballPlayer/dataset.yaml'  # 데이터셋 설정 파일 경로
epochs = 10  # 학습 에포크
image_size = 1920  # 이미지 크기 (성능 및 정확도 조절 위해 변경 가능)
batch_size = 5  # 배치 크기
weights = 'yolov8n.pt'  # 사전 학습된 모델 경로 (사용하지 않을 경우 'None'으로 설정)
project = '/content/drive/MyDrive/FootballPlayer/TrainingResults'  # 학습 결과 저장 경로
name = 'footballDetection'  # 학습 런 이름

# 데이터셋 로딩
dataset = LoadImagesAndLabels(data_path, img_size=image_size, augment=True, rect=True)  # 데이터 증강 및 박스 형식 설정

# YOLOv8 모델 생성
model = YOLOv8(weights=weights)  # 사전 학습된 모델 로드

# 학습 시작
train_detector(model, dataset, device=device, epochs=epochs, batch_size=batch_size, project=project, name=name)

# 학습 완료 메시지 출력
print("Training completed! Your trained model can be found in", project)
