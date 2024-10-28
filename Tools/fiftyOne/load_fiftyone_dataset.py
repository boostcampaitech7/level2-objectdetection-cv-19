import json
import fiftyone as fo
import os

# 1. 데이터셋 디렉토리 경로 설정
DATASET_DIR = "/data/ephemeral/home/origin_new/origin_new"  # 실제 경로로 변경하세요
IMAGE_DIR = os.path.join(DATASET_DIR, "data")
SAMPLES_JSON_PATH = os.path.join(DATASET_DIR, "samples.json")
METADATA_JSON_PATH = os.path.join(DATASET_DIR, "metadata.json")  # 필요 시 사용

# 2. JSON 파일 로드
with open(SAMPLES_JSON_PATH, 'r') as f:
    data = json.load(f)

# 필요 시 metadata.json 로드 (현재 사용하지 않음)
# with open(METADATA_JSON_PATH, 'r') as f:
#     metadata = json.load(f)
# metadata 처리 로직 추가

# 3. FiftyOne 데이터셋 생성 (이미 존재하면 로드)
dataset_name = "origin_new"
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = fo.Dataset(dataset_name)

# 4. 샘플 추가
for sample in data['samples']:
    # 파일 경로 설정 (상대 경로 "data/3480.jpg"를 절대 경로로 변환)
    filepath = os.path.join(DATASET_DIR, sample['filepath'])  # 예: "/path/to/origin_new/data/3480.jpg"
    
    # 파일이 존재하는지 확인
    if not os.path.exists(filepath):
        print(f"파일이 존재하지 않습니다: {filepath}")
        continue
    
    # FiftyOne 샘플 생성
    fo_sample = fo.Sample(filepath=filepath)
    
    # Ground Truth Detections 추가
    if 'ground_truth' in sample:
        gt_detections = fo.Detections(detections=[])
        for det in sample['ground_truth']['detections']:
            detection = fo.Detection(
                label=det['label'],
                bounding_box=det['bounding_box']  # [left, top, width, height]
            )
            gt_detections.detections.append(detection)
        fo_sample['ground_truth'] = gt_detections
    
    # Predictions Detections 추가
    if 'predictions' in sample:
        pred_detections = fo.Detections(detections=[])
        for det in sample['predictions']['detections']:
            detection = fo.Detection(
                label=det['label'],
                bounding_box=det['bounding_box'],
                confidence=det.get('confidence', 1.0)  # confidence 값이 없으면 1.0으로 설정
            )
            pred_detections.detections.append(detection)
        fo_sample['predictions'] = pred_detections
    
    # 샘플을 데이터셋에 추가 (이미 존재하면 업데이트)
    dataset.add_sample(fo_sample)

# 5. FiftyOne 앱 실행 (시각화)
session = fo.launch_app(dataset)

# 6. 필요 시 추가적인 시각화 옵션 설정 (예: 레이어 설정)
# 예를 들어, ground_truth와 predictions를 동시에 보려면:
session.view["predictions"] = dataset.to_view().add_stage(
    fo.BatchRename("ground_truth", "predictions")
)

# 7. 세션 유지 (원하는 경우)
# session.wait()  # 세션을 유지하려면 주석 해제
# session.close() # 세션을 종료하려면 주석 해제
