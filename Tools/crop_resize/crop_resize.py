import cv2
import json
import os
from tqdm import tqdm  # 진행 상태를 시각적으로 표시하기 위해 사용
import numpy as np

# JSON 파일에서 bbox 정보를 가져오는 함수
def load_bboxes_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    bboxes = data['annotations']  # [x, y, width, height] 형식의 리스트로 되어 있다고 가정
    return bboxes, data

# 작은 bbox를 찾아 크롭하고 리사이즈하는 함수
def process_bbox(image, bbox, scale_factor=1.5):
    x, y, width, height = map(int, bbox)
    
    # 이미지 경계 확인 및 조정
    img_height, img_width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    width = min(width, img_width - x)
    height = min(height, img_height - y)

    # 유효한 bounding box인지 확인
    if width <= 0 or height <= 0:
        # 크기 조정 후에도 유효하지 않으면 스킵
        return None, None

    # bbox 영역 크롭
    cropped = image[y:y+height, x:x+width]
    
    # 크롭된 이미지가 비어 있는지 확인
    if cropped.size == 0:
        print(1)
        return None, None
    
    # 크롭된 이미지를 리사이즈
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized = cv2.resize(cropped, new_size)
    
    # 새로운 bbox 위치 계산
    new_bbox = [x, y, int(width * scale_factor), int(height * scale_factor)]
    return resized, new_bbox

# 원래 이미지에 새로운 이미지를 붙여넣는 함수
def paste_to_original_image(original_image, resized_image, new_bbox):
    x, y, new_width, new_height = new_bbox
    # 원본 이미지의 크기 확인
    img_height, img_width = original_image.shape[:2]

    # 이미지 경계를 벗어나는지 확인하고 조정
    new_width = min(new_width, img_width - x)
    new_height = min(new_height, img_height - y)
    
    # 붙여넣을 위치와 크기 조정
    resized_image = resized_image[:new_height, :new_width]
    
    # 원래 이미지에 resized_image 붙여넣기
    original_image[y:y+new_height, x:x+new_width] = resized_image
    return original_image

# 메인 함수
def process_images_in_folder(folder_path, json_file_path, output_json_file_path, output_folder_path):
    # 결과를 저장할 폴더 생성
    os.makedirs(output_folder_path, exist_ok=True)
    
    # train 폴더에 있는 모든 이미지 파일 처리
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg'))]
    
    # JSON 파일에서 bbox 정보 로드
    if not os.path.exists(json_file_path):
        print(f"Warning: {json_file_path} does not exist.")
        return
    
    bboxes, json_data = load_bboxes_from_json(json_file_path)
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, image_file)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: {image_path} could not be loaded. Skipping.")
            continue
        
        resize_id = []
        # 각 bbox에 대해 처리
        for ann in json_data['annotations']:
            # bbox가 너무 작은 경우에만 처리
            if ann['bbox'][2] * ann['bbox'][3] < 10000:  # 넓이가 10000보다 작은 경우
                resized_image, new_bbox = process_bbox(image, ann['bbox'], scale_factor=2.5)
                if resized_image is None:
                    continue  # 이미지가 비어 있으면 건너뛰기
                else:
                    resize_id.append(ann['image_id'])
                # 원래 이미지에 붙여넣기
                image = paste_to_original_image(image, resized_image, new_bbox)
                # bbox 업데이트
                ann['bbox'] = new_bbox
        
        # 결과 이미지 저장
        output_image_path = os.path.join(output_folder_path, image_file)
        cv2.imwrite(output_image_path, image)
    
    # JSON 파일 덮어쓰기
    with open(output_json_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(resize_id)
    print(f"모든 이미지 처리가 완료되었습니다. 결과는 {output_folder_path}에 저장되었고, bbox 정보는 {json_file_path}에 업데이트되었습니다.")

# 실행 예시
train_folder_path = '/data/ephemeral/home/sr_dataset/train'
json_file_path = '/data/ephemeral/home/sr_dataset/train.json'  # bbox JSON 파일 경로
output_json_file_path = '/data/ephemeral/home/sr_dataset/resize_train.json'
output_folder_path = '/data/ephemeral/home/sr_dataset/output_train'

process_images_in_folder(train_folder_path, json_file_path, output_json_file_path, output_folder_path)
