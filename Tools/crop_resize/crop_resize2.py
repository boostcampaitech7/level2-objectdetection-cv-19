import json
import cv2
import os
from tqdm import tqdm

def load_coco_annotations(json_path):
    # COCO 스타일 JSON 파일을 로드
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_image_annotations(image_id, coco_data):
    # 주어진 image_id에 해당하는 모든 annotation 검색
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    return annotations

def scale_bbox(bbox, scale_factor):
    # 주어진 scale factor에 따라 bbox 크기를 변경
    x, y, width, height = bbox
    new_width = width * scale_factor
    new_height = height * scale_factor
    new_x = x - (new_width - width) / 2  # bbox 중심을 유지하며 크기 변경
    new_y = y - (new_height - height) / 2
    scaled_bbox = [new_x, new_y, new_width, new_height]
    # 크기 제한
    return limit_bbox_to_bounds(scaled_bbox)

# def limit_bbox_to_bounds(bbox, image_width=1024, image_height=1024):
#     # bbox가 이미지 크기를 넘지 않도록 제한
#     x, y, width, height = bbox
#     x = max(0, min(x, image_width))
#     y = max(0, min(y, image_height))
#     width = min(width, image_width - x)
#     height = min(height, image_height - y)
#     return [x, y, width, height]

def limit_bbox_to_bounds(bbox, image_width=1024, image_height=1024):
    x, y, width, height = bbox
    center_x = x + width / 2
    center_y = y + height / 2

    # bbox가 이미지 경계를 넘어가지 않도록 크기 조정
    if width > image_width:
        width = image_width
    if height > image_height:
        height = image_height

    # 중심을 기준으로 bbox 크기 조정
    x = max(0, min(center_x - width / 2, image_width - width))
    y = max(0, min(center_y - height / 2, image_height - height))

    return [x, y, width, height]


def resize_and_paste(image, bbox, scale_factor):
    # bbox를 사용하여 이미지 일부를 크롭
    x, y, width, height = map(int, bbox)
    cropped_image = image[y:y+height, x:x+width]
    
    # 크롭된 이미지를 리사이즈
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(cropped_image, (new_width, new_height))
    
    # 붙여넣을 위치 계산
    paste_x = int(x - (new_width - width) / 2)
    paste_y = int(y - (new_height - height) / 2)
    
    # 경계 검사 및 조정 (이미지 크기 범위를 넘지 않도록)
    paste_x = max(0, min(paste_x, image.shape[1] - new_width))
    paste_y = max(0, min(paste_y, image.shape[0] - new_height))
    
    # 리사이즈된 이미지를 원본 이미지에 붙여넣기
    if paste_x + new_width <= image.shape[1] and paste_y + new_height <= image.shape[0]:
        image[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_image
    else:
        print(f"Skipping resize and paste as it exceeds image bounds for bbox {bbox}")
    
    return image

def process_images(coco_data, images_dir, output_dir, scale_factor=1.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(images_dir, file_name)
        
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            continue
        
        # 해당 image_id에 대한 annotation 검색
        annotations = get_image_annotations(image_id, coco_data)
        
        # 해당 이미지의 각 annotation(bbox) 처리
        for annotation in annotations:
            bbox = annotation['bbox']
            bbox_width, bbox_height = bbox[2], bbox[3]
            
            # bbox 면적이 10,000 이하일 때만 처리
            if bbox_width * bbox_height <= 10000:
                # bbox를 scale하고 크롭된 이미지를 리사이즈
                new_bbox = scale_bbox(bbox, scale_factor)
                image = resize_and_paste(image, new_bbox, scale_factor)
                
                # annotation의 bbox 업데이트
                annotation['bbox'] = new_bbox

        # 수정된 이미지 저장
        output_path = os.path.join(output_dir, file_name)
        save_success = cv2.imwrite(output_path, image)

        # # 저장 여부 확인
        # if save_success:
        #     print(f"Saved modified image to {output_path}")
        # else:
        #     print(f"Failed to save image {output_path}. Check if the path is correct and writable.")
    
    
    # 수정된 annotation 데이터를 다시 JSON 파일로 저장
    with open(os.path.join(output_dir, 'modified_annotations.json'), 'w') as f:
        json.dump(coco_data, f)

if __name__ == "__main__":
    # JSON 파일 및 이미지 디렉토리 경로 설정
    json_path = "/data/ephemeral/home/sr_dataset/train.json"
    images_dir = "/data/ephemeral/home/sr_dataset/"
    output_dir = "/data/ephemeral/home/sr_dataset/output_train"
    
    # COCO 데이터셋 annotation 로드
    coco_data = load_coco_annotations(json_path)
    
    # 이미지 처리 및 결과 저장
    process_images(coco_data, images_dir, output_dir, scale_factor=1.5)
