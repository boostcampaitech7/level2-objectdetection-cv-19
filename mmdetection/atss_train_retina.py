# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/atss/atss_retina_coco.py')
cfg.work_dir = './work_dirs/SR_retina_atss_trash'

# cfg = Config.fromfile('./configs/atss/atss_swin_coco.py')
# cfg.work_dir = './work_dirs/swin_atss_trash'

train_root='/data/ephemeral/home/sr_dataset/output_train/'
test_root = '/data/ephemeral/home/sr_dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = train_root
cfg.data.train.ann_file = train_root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = test_root
cfg.data.test.ann_file = test_root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]

cfg.model.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# dataset 확인
datasets[0]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

