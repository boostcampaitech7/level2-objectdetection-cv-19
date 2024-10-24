#!/usr/bin/env python
# coding: utf-8

# # Transform the Result SR images to add to dataset
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from datetime import datetime
import random


# ## Make X2 annotation

# ### Check annotation size

# Dataset path
dataDir = '/data/ephemeral/home/sr_dataset'
annotation_path = '/data/ephemeral/home/dataset/train.json'

# ### Make SR dataset annotation file

x2_annotation_path = '/data/ephemeral/home/sr_dataset/train_x2_SR.json'

# Read json
with open(annotation_path, 'r') as file:
    data = json.load(file)

# Modify bbox values
for ann in data['annotations']:
    ann['bbox'][0] *= 2
    ann['bbox'][1] *= 2
    ann['bbox'][2] *= 2
    ann['bbox'][3] *= 2

# Store new json file
with open(x2_annotation_path, 'w') as file:
    json.dump(data, file, indent=2)


# ## Divide the image into quarters

# Dataset path
dataDir = '/data/ephemeral/home/rd_dataset'
annotation_path = '/data/ephemeral/home/rd_dataset/train_x2_SR.json'
original_anno_path = '/data/ephemeral/home/dataset/train.json'

# saved path
subimgs_path = '/data/ephemeral/home/rd_dataset/subimages'
updated_annotation_path = '/data/ephemeral/home/rd_dataset/train_x2_SR_Random_images.json'

def update_annotations_for_subimage(annotations, subimg_info, img_id):
    updated_annotations = []
    x_offset, y_offset, subimg_width, subimg_height = subimg_info

    for ann in annotations:
        x, y, width, height = ann['bbox']

        # BBox가 subimg 영역과 겹치는지 확인
        if (x + width > x_offset and x < x_offset + subimg_width and
            y + height > y_offset and y < y_offset + subimg_height):
            
            # Update BBox coordinate
            new_x = max(x - x_offset, 0)
            new_y = max(y - y_offset, 0)
            width = min(width, x+width - x_offset)
            height = min(height, y+height - y_offset)

            updated_ann = ann.copy()
            updated_ann['bbox'] = [new_x, new_y, width, height]
            updated_ann['image_id'] = img_id
            updated_annotations.append(updated_ann)
    
    return updated_annotations

    
def generate_random_crop_regions(image, crop_width=1024, crop_height=1024, num_regions=2):
    """
    이미지를 입력받아 랜덤한 크롭 영역 좌표를 생성하는 함수.
    
    :param image: PIL 이미지 객체
    :param crop_width: 크롭할 영역의 너비 (기본값: 1024)
    :param crop_height: 크롭할 영역의 높이 (기본값: 1024)
    :param num_regions: 생성할 크롭 영역의 수 (기본값: 2)
    :return: 크롭 영역 리스트 [(left, top, right, bottom), ...]
    """
    img_width, img_height = image.size
    subimages = []

    for _ in range(num_regions):

        # 크롭할 영역의 좌상단 (left, top) 좌표를 랜덤하게 설정
        left = random.randint(0, img_width - crop_width)
        top = random.randint(0, img_height - crop_height)

        # # 우하단 (right, bottom) 좌표 계산
        # right = left + crop_width
        # bottom = top + crop_height

        # subimages 리스트에 (left, top, right, bottom) 형식으로 추가
        subimages.append((left, top, crop_width, crop_height))

    return subimages


# Read annotation file
with open(annotation_path, 'r') as file:
    data = json.load(file)

# Define new images and annotations
new_images = []
new_annotations = []
new_img_id = max([img['id'] for img in data['images']]) + 1

# Load images
coco = COCO(annotation_path)
for idx in os.listdir(os.path.join(dataDir,'train')):
    img = coco.loadImgs(int(idx.split('_')[0]))[0]
    I = Image.open('{}/{}_x2_SR.png'.format(dataDir, img['file_name'].split('.')[0]))
    img_width, img_height = I.size

    # annotation ID
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)

    # 이미지를 4등분하는 영역 정의
    # subimages = [
    #     (0, 0, img_width // 2, img_height // 2),  # 상단 왼쪽
    #     (img_width // 2, 0, img_width // 2, img_height // 2),  # 상단 오른쪽
    #     (0, img_height // 2, img_width // 2, img_height // 2),  # 하단 왼쪽
    #     (img_width // 2, img_height // 2, img_width // 2, img_height // 2)  # 하단 오른쪽
    # ]
    subimages = generate_random_crop_regions(I, img_width // 2, img_height // 2, 2)
    # print(subimages)

    # Update annotations for each partial image
    for i, subimg_info in enumerate(subimages):
        updated_anns = update_annotations_for_subimage(anns, subimg_info, new_img_id)

        # Draw X2 bounding box
        x_offset, y_offset, subimg_width, subimg_height = subimg_info
        subimg = I.crop((x_offset, y_offset, x_offset + subimg_width, y_offset + subimg_height))
        # print((x_offset, y_offset, x_offset + subimg_width, y_offset + subimg_height))
        # plt.imshow(subimg)
        plt.axis('off')
        
        ax = plt.gca()
        for ann in updated_anns:
            bbox = ann['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        # plt.show()

        subimg_filename = '{}_{}_x2_SR.png'.format(img['file_name'].split('.')[0], i)
        # annotation file updated
        if updated_anns:
            new_img = {
                "width": subimg_width,
                "height": subimg_height,
                "file_name": subimg_filename,
                "license": 0,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "id": new_img_id
            }
            new_images.append(new_img)
            new_annotations.extend(updated_anns)

            # bbox가 있는 경우만 subimg 저장 
            subimg.save(os.path.join(subimgs_path, subimg_filename))
        
            new_img_id += 1

# 추가는 train.json 파일로 해야함
with open(original_anno_path, 'r') as file:
    original_data = json.load(file)

original_data['images'].extend(new_images)
original_data['annotations'].extend(new_annotations)

with open(updated_annotation_path, 'w') as file:
    json.dump(original_data, file, indent=2)

