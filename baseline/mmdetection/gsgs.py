import os.path as osp
import json
import pandas as pd
from pycocotools.coco import COCO
from collections import defaultdict

# 작업 디렉토리 설정
work_dir = "C:/Users/young/Desktop/부스트캠프/프로젝트-02/백업/origin_new"  # 실제 경로로 변경하세요
test_fixed_ann_file = osp.join(work_dir, 'test_fixed_annotations.json')  # 실제 주석 파일 경로로 변경

# Confidence threshold 설정
confidence_threshold = 0.3

# 1. COCO 파일 로드하여 이미지 정보 가져오기
coco = COCO(test_fixed_ann_file)
img_ids = coco.getImgIds()
img_info = coco.loadImgs(img_ids)

# 2. 이미지 ID를 file_name으로 매핑
id_to_filename = {img['id']: img['file_name'] for img in img_info}

# 3. 테스트 결과 로드
test_results_file = osp.join(work_dir, 'test_results.bbox.json')  # 실제 테스트 결과 파일 경로로 변경
with open(test_results_file, 'r') as f:
    results = json.load(f)

# 4. 이미지별로 예측 결과를 그룹화
imgid_to_results = defaultdict(list)
for res in results:
    imgid_to_results[res['image_id']].append(res)

# 5. Confidence threshold를 적용하여 예측 필터링 및 제출 문자열 생성
prediction_strings = []
file_names = []

for img_id in img_ids:
    preds = imgid_to_results.get(img_id, [])
    prediction_string = ''
    for pred in preds:
        score = pred['score']
        if score >= confidence_threshold:  # Confidence threshold 적용
            cat_id = pred['category_id'] - 1  # COCO의 category_id는 1부터 시작하므로 0부터 시작하도록 조정
            bbox = pred['bbox']  # [x, y, width, height]
            # COCO 형식의 bbox는 [x, y, width, height], 이를 [x1, y1, x2, y2]로 변환
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            prediction_string += f"{cat_id} {score:.8f} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f} "
    
    # Confidence threshold 적용 후 예측이 있는 경우에만 추가
    if prediction_string:  
        prediction_strings.append(prediction_string.strip())
        file_names.append(id_to_filename[img_id])

# 6. 제출용 데이터프레임 생성
submission = pd.DataFrame({
    'PredictionString': prediction_strings,
    'image_id': file_names
})

# 7. 제출 CSV 파일 저장
submission_file = osp.join(work_dir, 'submission.csv')
submission.to_csv(submission_file, index=False)
print(f"제출 파일이 저장되었습니다: {submission_file}")

# 8. 제출 파일의 첫 몇 개 항목 표시
print(submission.head())
