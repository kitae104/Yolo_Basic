# 자동 라벨링을 위한 코드
import os
import cv2
from glob import glob
from ultralytics import YOLO

image_dir = '../datasets/abnormal/ochang_dataset_2023/auto_annotated/train/images'
label_dir = f'{image_dir}/../labels'
visual_dir = f'{image_dir}/../annotated'

os.makedirs(label_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)


# auto annotation
def auto_annotation():
    weight = 'weights/yolov8x-pose.pt'
    auto_annotate(weight=weight, device='0', conf=0.3)


def auto_annotate(weight, device, conf=0.1):
    model = YOLO(weight)

    image_list = glob(f'{image_dir}/*.jpg')

    for image_path in image_list:
        results = model.predict(image_path, device=device, conf=conf, stream=True)
        for result in results:
            class_ids = result.boxes.cls.int().tolist()  # noqa
            if len(class_ids):
                image_stem = os.path.basename(result.path).rsplit(".", 1)[0]

                annotated = result.plot(line_width=2)
                cv2.imwrite(f'{visual_dir}/{image_stem}.jpg', annotated)

                boxes = result.boxes.xyxyn  # Boxes object for bbox outputs
                kpts = result.keypoints.xyn
                with open(f'{label_dir}/{image_stem}.txt', 'w') as f:
                    for i in range(len(kpts)):
                        kpt = kpts[i]
                        if len(kpt) == 0:
                            continue
                        box = map(str, boxes[i].reshape(-1).tolist())
                        keypoint = map(str, kpts[i].reshape(-1).tolist())
                        f.write(f'{class_ids[i]} ' + ' '.join(box) + ' '.join(keypoint) + '\n')


if __name__ == '__main__':
    auto_annotation()
