import json

# JSON 파일 읽기
with open('/data/ephemeral/home/rd_dataset/train_x2_SR_Random_images.json', 'r') as f:
    data = json.load(f)

# 사용된 ID 추적 및 중복 ID 재할당
used_ids = set()
for annotation in data['annotations']:
    ann_id = annotation['id']
    if ann_id in used_ids:
        # 중복된 경우, 새로운 고유 ID 할당
        new_id = max(used_ids) + 1
        annotation['id'] = new_id
    used_ids.add(annotation['id'])

# 수정된 JSON 파일 저장
with open('/data/ephemeral/home/sr_dataset/dele_train_x2_SR_Random_images_fixed.json', 'w') as f:
    json.dump(data, f)

print("어노테이션 ID를 수정하고 'train_x2_SR_4images_fixed.json'에 저장했습니다.")
