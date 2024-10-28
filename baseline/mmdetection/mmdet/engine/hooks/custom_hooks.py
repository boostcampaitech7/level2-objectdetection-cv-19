from mmengine.hooks import Hook
from mmdet.apis import inference_detector
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import os

class TopMisclassifiedImagesHook(Hook):
    def __init__(self, dataset, work_dir, classes, top_k=20, score_thr=0.3):
        self.dataset = dataset
        self.work_dir = work_dir
        self.classes = classes
        self.top_k = top_k
        self.score_thr = score_thr

    def after_val_epoch(self, runner):
        misclassified = []
        for idx, data in enumerate(self.dataset):
            img_path = os.path.join(data['data_prefix']['img'], data['img_info']['filename'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image {img_path} could not be read.")
                continue

            # Ground Truth 가져오기
            gt_labels = data.get('gt_labels', [])
            gt_bboxes = data.get('gt_bboxes', [])  # gt_bboxes가 없을 경우 빈 리스트

            # 모델 추론
            results = inference_detector(runner.model, img)

            # 결과 처리 (bbox 결과 가정)
            if isinstance(results, tuple):
                bbox_result = results[0]
            else:
                bbox_result = results

            # 예측된 클래스 라벨 수집
            pred_labels = []
            for class_id, bboxes in enumerate(bbox_result, start=1):
                for bbox in bboxes:
                    pred_labels.append(class_id)

            # Ground Truth와 예측된 라벨 비교
            gt_set = set(gt_labels.tolist()) if isinstance(gt_labels, np.ndarray) else set(gt_labels)
            pred_set = set(pred_labels)
            misclass = not gt_set.issubset(pred_set)

            if misclass:
                # 오분류 정도를 나타내는 오류 점수 계산 (대칭 차집합의 크기)
                error_score = len(gt_set.symmetric_difference(pred_set))
                misclassified.append((error_score, img_path, img, results, gt_bboxes, gt_labels))

        # 오류 점수를 기준으로 내림차순 정렬
        misclassified = sorted(misclassified, key=lambda x: x[0], reverse=True)
        top_misclassified = misclassified[:self.top_k]

        # 상위 오분류된 이미지를 WandB에 로그
        for error_score, img_path, img, results, gt_bboxes, gt_labels in top_misclassified:
            # 이미지에 예측 결과와 Ground Truth를 그려줍니다.
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Ground Truth 바운딩 박스 그리기
            for bbox, label in zip(gt_bboxes, gt_labels):
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"GT: {self.classes[label - 1]}", color='green', fontsize=12, backgroundcolor='black')

            # 예측된 바운딩 박스 그리기
            for class_id, bboxes in enumerate(results, start=1):
                for bbox in bboxes:
                    score = bbox[-1]
                    if score < self.score_thr:
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f"{self.classes[class_id - 1]}: {score:.2f}", color='yellow', fontsize=12, backgroundcolor='black')

            ax.axis('off')

            # 시각화된 이미지를 파일로 저장
            vis_path = os.path.join(self.work_dir, f"misclassified_{os.path.basename(img_path)}.png")
            fig.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # WandB에 시각화된 이미지 로그
            runner.logger.experiment.log({
                "Top Misclassified Images": wandb.Image(vis_path, caption=f"Path: {img_path} | Error Score: {error_score}")
            })

        print(f"WandB에 상위 {self.top_k}개의 오분류 이미지를 로그했습니다.")