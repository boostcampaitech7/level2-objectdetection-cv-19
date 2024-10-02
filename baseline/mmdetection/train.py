import argparse
import time
import logging
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.utils import get_device

# Define the main function
def main(args):
    # Set up logging
    logging.basicConfig(filename=args.log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Log all hyperparameters at the beginning
    logging.info("Hyperparameters: %s", vars(args))

    logging.info('Starting training...')

    # Start the timer
    start_time = time.time()

    # Define the class labels
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # Load the configuration file
    cfg = Config.fromfile(args.config_file)

    # Set dataset root path
    root = args.dataset_root

    # Modify the dataset configuration
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json'  # Training JSON annotations
    cfg.data.train.pipeline[2]['img_scale'] = (args.img_width, args.img_height)  # Image resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'  # Test JSON annotations
    cfg.data.test.pipeline[1]['img_scale'] = (args.img_width, args.img_height)  # Image resize

    cfg.data.samples_per_gpu = args.samples_per_gpu  # Batch size per GPU

    cfg.seed = args.seed
    cfg.gpu_ids = [args.gpu_id]
    cfg.work_dir = args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    # Enable logging for loss and accuracy in MMDetection
    cfg.log_config = dict(
        interval=args.log_interval,
        hooks=[
            dict(type='TextLoggerHook'),  # Log to console
            dict(type='TensorboardLoggerHook')  # Optionally log to TensorBoard
        ]
    )

    # Build the detector model and initialize weights
    model = build_detector(cfg.model)
    model.init_weights()

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Train the model
    train_detector(model, datasets[0], cfg, distributed=False, validate=args.validate)

    # End the timer
    end_time = time.time()
    total_time = end_time - start_time

    # Log the total training time
    logging.info(f'Total training time: {total_time:.2f} seconds')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model on custom dataset")
    
    # Add arguments for hyperparameters
    parser.add_argument('--config_file', type=str, default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 
                        help='Path to the config file')
    parser.add_argument('--dataset_root', type=str, default='../../dataset/', 
                        help='Root directory of the dataset')
    parser.add_argument('--img_width', type=int, default=512, 
                        help='Width of the input image after resizing')
    parser.add_argument('--img_height', type=int, default=512, 
                        help='Height of the input image after resizing')
    parser.add_argument('--samples_per_gpu', type=int, default=4, 
                        help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=2022, 
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID to use')
    parser.add_argument('--work_dir', type=str, default='./work_dirs/faster_rcnn_r50_fpn_1x_trash', 
                        help='Working directory to save logs and models')
    parser.add_argument('--log_file', type=str, default='training.log', 
                        help='File path for saving training logs')
    parser.add_argument('--log_interval', type=int, default=50, 
                        help='Log interval for printing loss and metrics during training')
    parser.add_argument('--validate', action='store_true', 
                        help='Whether to validate the model during training')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
