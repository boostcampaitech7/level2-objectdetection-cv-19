# Object Detection for Waste Segregation

## 개요
오늘날 우리는 대량 생산과 소비로 인한 환경 문제에 직면해 있습니다. 매립지 부족과 같은 쓰레기 문제는 심각한 사회적 과제로 대두되고 있으며, 이를 해결하기 위한 방안으로 분리수거의 중요성이 강조되고 있습니다. 적절히 분리수거된 쓰레기는 재활용 자원으로 활용될 수 있지만, 잘못 배출되면 매립 또는 소각이 되어 환경에 부담을 줄 수 있습니다.

본 프로젝트는 쓰레기 사진에서 객체를 탐지하여 쓰레기 종류를 분류하고, 분리수거 과정을 자동화하는 모델을 만드는 것을 목표로 합니다. 제공된 데이터셋은 COCO 포맷으로, 일반 쓰레기, 플라스틱, 종이, 유리 등 10가지의 쓰레기 종류가 포함된 이미지들로 구성되어 있습니다. 이 모델이 개발된다면 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린이들의 분리수거 교육에 활용될 수 있습니다.

## 데이터
- **Input**: 모델의 입력으로 쓰레기 객체가 포함된 이미지가 사용되며, 학습 시에는 bbox 정보(좌표, 카테고리)를 포함한 COCO 포맷의 어노테이션이 사용됩니다.
- **Output**: 모델은 bbox 좌표, 카테고리, 그리고 예측 점수(score)를 반환하며, 이를 기반으로 csv 포맷의 제출 파일을 생성합니다.

## 평가 방법
모델 성능은 Test set의 mAP50(Mean Average Precision) 기준으로 평가됩니다. 이는 Object Detection 분야에서 일반적으로 사용하는 성능 측정 방식으로, Ground Truth 박스와 Prediction 박스 간의 IoU(Intersection Over Union)가 0.5 이상일 때 True로 간주합니다.

![image](https://github.com/user-attachments/assets/e763b4cb-4be3-42e8-b8ca-856b424f2699)

![image](https://github.com/user-attachments/assets/d4fb7058-76fe-4974-9a55-7135c38c34c9)

![image](https://github.com/user-attachments/assets/e7e823ed-880c-4016-8d8f-68043697974f)

## 앙상블 전략
본 프로젝트에서는 Salience-DETR, Co-DETR, ATSS, YOLO 모델을 개별적으로 학습 및 평가한 후, 예측 결과를 앙상블하여 최적의 탐지 성능을 달성했습니다. 이를 위해 다양한 앙상블 기법(WBF 등)을 적용하여 각 모델의 장점을 결합하고 성능을 극대화하였습니다.

## 파일 구조
```plaintext
.
├── README.md                  # 프로젝트 개요 및 설명
├── Tools                      # 데이터 전처리 및 후처리 도구 모음
│   ├── crop_resize            # 이미지 크롭 및 리사이즈 관련 도구
│   ├── fiftyOne               # 데이터셋 분석 및 시각화를 위한 파일
│   └── super_resolution       # 이미지 해상도 개선 관련 도구
├── baseline                   # 모델 학습을 위한 기본 설정 파일
│   ├── Salience-DETR          # Salience-DETR 모델 설정 파일
│   ├── detectron2             # Detectron2 모델 설정 파일
│   ├── faster_rcnn            # Faster R-CNN 모델 설정 파일
│   ├── mmdetection            # mmdetection 모델 설정 파일
│   └── requirements.txt       # 필수 패키지 목록
└── notebooks                  # Jupyter 노트북 모음
    ├── dino_test_tta.ipynb    # DINO 모델 테스트 타임 어그멘테이션 코드
    ├── dino_train.ipynb       # DINO 모델 학습 코드
    ├── wbf_ensemble.ipynb     # 기본 WBF 앙상블 코드
    ├── wbf_ensemble_max.ipynb # 최대 가중치 적용 WBF 앙상블 코드
    ├── yolov11.ipynb          # YOLOv11 모델 실험 코드
    └── 앙상블 시각화.ipynb      # 앙상블 결과 시각화 코드
```
이 리포지토리에서는 쓰레기 객체 탐지 대회의 성능을 높이기 위해 **앙상블 방식**을 채택하였습니다. 이를 위해 Salience-DETR, Co-DETR, ATSS, YOLO와 같은 네 가지 서로 다른 모델을 각각 실험하고, 다양한 방법으로 결과를 융합하여 최적의 예측 성능을 도출하는 것을 목표로 하였습니다. 리포지토리에는 각 모델의 구현 및 실험 파일과, 모델 결과를 앙상블하기 위한 코드가 포함되어 있습니다.
### 주요 파일 및 디렉토리 설명

- **Tools/**: 데이터 전처리 및 후처리에 사용되는 다양한 도구 모음입니다.
  - **crop_resize/**: 이미지의 크롭 및 리사이즈 기능을 제공합니다.
  - **fiftyOne/**: 데이터셋 분석과 시각화를 위해 **FiftyOne** 라이브러리를 활용하는 파일입니다.
  - **super_resolution/**: 이미지의 해상도를 개선하는 데 필요한 도구들이 포함됩니다.

- **baseline/**: 다양한 객체 탐지 모델의 설정 파일과 기본 학습 환경을 제공합니다.
  - **Salience-DETR/**: 쓰레기 객체 탐지를 위한 **Salience-DETR** 모델의 설정 파일이 포함되어 있습니다.
  - **detectron2/**: **Detectron2** 기반 객체 탐지 모델 설정 파일을 포함합니다.
  - **faster_rcnn/**: **Faster R-CNN** 모델의 설정 파일이 포함되어 있습니다.
  - **mmdetection/**: **MMDetection 프레임워크** 기반 설정 디렉토리로, Co-DETR 및 ATSS 모델 설정 파일이 포함되어 있습니다.
    - **Co-DETR**: **Conditional-DETR** 모델로, DETR(Detection Transformer)의 확장 버전입니다. 조건부 앵커를 사용하여 위치 정보를 개선하고, 더욱 정확한 객체 탐지를 수행합니다.
    - **ATSS**: **Adaptive Training Sample Selection** 모델로, 객체 크기와 관계없이 효과적인 탐지를 위해 학습 시 적응적으로 샘플을 선택하는 방식입니다.
  - **requirements.txt**: 프로젝트 실행에 필요한 패키지 목록이 정리되어 있습니다.

- **notebooks/**: 다양한 실험과 앙상블 과정을 수행하는 **Jupyter 노트북** 모음입니다.
  - **dino_train.ipynb** 및 **dino_test_tta.ipynb**: **DINO 모델**의 학습과 **테스트 타임 어그멘테이션**(TTA)을 위한 노트북입니다.
  - **wbf_ensemble.ipynb** 및 **wbf_ensemble_max.ipynb**: **Weighted Boxes Fusion (WBF)** 앙상블 기법을 통해 예측 결과를 개선하는 코드입니다.
  - **yolov11.ipynb**: **YOLOv11** 모델을 사용하여 실험 및 객체 탐지를 수행하는 코드입니다.
  - **앙상블 시각화.ipynb**: 다양한 모델의 예측 결과를 앙상블하여 시각화하고, 성능을 직관적으로 확인할 수 있도록 합니다.

## 설치 및 실행 안내

이 프로젝트는 여러 가지 모델(Salience-DETR, Co-DETR, ATSS, YOLO 등)을 포함하고 있으며, 각 모델의 환경 설정과 학습, 테스트 과정이 서로 상이합니다. 이에 따라 개별 모델에 맞춘 설정과 실행 방법은 각 디렉토리와 노트북 파일에 포함된 주석과 코드 설명을 참조해주시기 바랍니다.

모델의 학습과 테스트 절차에 대한 세부적인 설명은 각 **Jupyter 노트북**에서 순차적으로 안내하고 있으며, **필요한 패키지는 `baseline/requirements.txt`** 파일에 정리되어 있습니다. 이를 통해 필요한 패키지를 설치한 후, 각 모델의 실험 파일을 단계적으로 실행하여 실험을 재현할 수 있습니다.

모델별 실행 과정에 대한 추가적인 설명이나 가이드가 필요하신 경우, 리포지토리 내 주석을 참조하시거나 이슈로 문의해 주시면 감사하겠습니다.


## 참고 자료
- [CO-DETR: End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2211.12860)
- [YOLOv11: An Overview of the Key Architectural Enhancements](https://arxiv.org/abs/2410.17725)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/en/latest/)
- [Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html)
