import contextlib
import datetime
import io
import logging
import math
import os
import sys
import time

import torch
import numpy as np
from terminaltables import AsciiTable

import util.utils as utils
from util.coco_eval import CocoEvaluator
from util.coco_utils import get_coco_api_from_dataset
from util.collate_fn import DataPrefetcher


def train_one_epoch_acc(
    model, optimizer, data_loader, epoch, print_freq=50, max_grad_norm=-1, accelerator=None
):
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("data_time", utils.SmoothedValue(fmt="{avg:.4f}"))
    metric_logger.add_meter("iter_time", utils.SmoothedValue(fmt="{avg:.4f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    prefetcher = DataPrefetcher(data_loader, accelerator.device)
    next_data_time = None
    data_start_time = time.perf_counter()
    images, targets = prefetcher.next()
    data_time = time.perf_counter() - data_start_time
    iter_start_time = time.perf_counter()
    for i in range(len(data_loader)):
        with accelerator.accumulate(model):
            # model forward
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # prefetch next batch data
            data_time = next_data_time if i > 0 else data_time
            if i < len(data_loader) - 1:
                data_start_time = time.perf_counter()
                images, targets = prefetcher.next()
                next_data_time = time.perf_counter() - data_start_time

            # backward propagation
            optimizer.zero_grad()
            accelerator.backward(losses)
            if accelerator.sync_gradients and max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if epoch == 0:
                lr_scheduler.step()

        # reduce losses over all GPUs for logging purposes
        with torch.no_grad():
            loss_dict_reduced = accelerator.reduce(loss_dict, reduction="mean")
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            logger.warning(f"Loss is {loss_value}, stopping training")
            logger.warning(loss_dict_reduced)
            sys.exit(1)

        # collect logging messages
        training_logs = {"loss": losses_reduced.item(), **loss_dict_reduced}
        training_logs.update({"lr": optimizer.param_groups[0]["lr"]})
        metric_logger.update(**training_logs)

        # update iter_time and data_time
        iter_time = time.perf_counter() - iter_start_time
        iter_start_time = time.perf_counter()
        metric_logger.update(**{"iter_time": iter_time, "data_time": data_time})

        # logging track
        if i % print_freq == 0:
            logger.info(get_logging_string(metric_logger, data_loader, i, epoch))
            training_logs = {k.replace("loss_", "loss/"): v for k, v in training_logs.items()}
            accelerator.log(training_logs, step=i + len(data_loader) * epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return metric_logger
@torch.no_grad()
def evaluate_acc(model, data_loader, epoch, accelerator=None):
    logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    # Initialize confusion matrix, detection count, and confidence record
    num_classes = len(coco.getCatIds())  # Get the total number of classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    confidence_matrix = [[[] for _ in range(num_classes)] for _ in range(num_classes)]  # Confidence scores storage
    category_det_nums = [0] * (num_classes + 1)  # Add +1 to avoid index errors
    
    # Dictionary to store samples with confidence below 0.5, categorized by actual class
    low_confidence_samples = {class_id: [] for class_id in range(num_classes)}
    
    # Track undetected samples (False Negatives) for each class
    undetected_samples = {class_id: 0 for class_id in range(num_classes)}

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        # Get model predictions
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]  # Move outputs to CPU
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
        evaluator_time = time.time()

        # Update confusion matrix and detection counts
        for target, output in zip(targets, outputs):
            gt_labels = target["labels"].cpu().numpy()  # Ground truth labels
            pred_labels = output["labels"].cpu().numpy()  # Predicted labels
            scores = output["scores"].cpu().numpy()  # Prediction confidence scores

            detected_gt = set()  # Track detected ground truth classes

            for gt, pred, score in zip(gt_labels, pred_labels, scores):
                if score > 0.5:  # Standard threshold for confusion matrix update
                    confusion_matrix[gt, pred] += 1
                    confidence_matrix[gt][pred].append(score)  # Record confidence score for each prediction
                    detected_gt.add(gt)
                else:
                    # If score is below 0.5, log the sample for its actual class
                    low_confidence_samples[gt].append((gt, pred, score))

            # Check for undetected ground truths (False Negatives)
            for gt in gt_labels:
                if gt not in detected_gt:
                    undetected_samples[gt] += 1  # Count False Negatives

            # Update category detection counts (dets)
            for pred in pred_labels:
                category_det_nums[pred] += 1  # Increase the detection count for the predicted class

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    coco_evaluator.synchronize_between_processes()

    # Accumulate predictions from all images
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    logger.info(redirect_string.getvalue())

    # Print category-wise evaluation results
    cat_names = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    table_data = [["class", "imgs", "gts", "dets", "recall", "ap"]]
    bbox_coco_eval = coco_evaluator.coco_eval["bbox"]
    
    for cat_idx, cat_name in enumerate(cat_names):
        cat_id = coco.getCatIds(catNms=cat_name)
        num_img_id = len(coco.getImgIds(catIds=cat_id))
        num_ann_id = len(coco.getAnnIds(catIds=cat_id))
        row_data = [cat_name, num_img_id, num_ann_id, category_det_nums[cat_id[0]]]
        row_data += [f"{bbox_coco_eval.eval['recall'][0, cat_idx, 0, 2].item():.3f}"]
        row_data += [f"{bbox_coco_eval.eval['precision'][0, :, cat_idx, 0, 2].mean().item():.3f}"]
        table_data.append(row_data)

    # Calculate and append mean results
    cat_recall = coco_evaluator.coco_eval["bbox"].eval["recall"][0, :, 0, 2]
    valid_cat_recall = cat_recall[cat_recall >= 0]
    mean_recall = valid_cat_recall.sum() / max(len(valid_cat_recall), 1)
    cat_ap = coco_evaluator.coco_eval["bbox"].eval["precision"][0, :, :, 0, 2]
    valid_cat_ap = cat_ap[cat_ap >= 0]
    mean_ap50 = valid_cat_ap.sum() / max(len(valid_cat_ap), 1)
    mean_data = ["mean results", "", "", "", f"{mean_recall:.3f}", f"{mean_ap50:.3f}"]
    table_data.append(mean_data)

    # Display results
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    logger.info("\n" + table.table)

    # Log confusion matrix
    logger.info("Confusion Matrix:")
    logger.info(f"\n{confusion_matrix}")

    

    return coco_evaluator

def get_logging_string(metric_logger, data_loader, i, epoch):
    MB = 1024 * 1024
    eta_seconds = metric_logger.meters["iter_time"].global_avg * (len(data_loader) - i)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    memory = torch.cuda.memory_allocated() / MB
    max_memory = torch.cuda.max_memory_allocated() / MB

    log_msg = f"Epoch: [{epoch}]  [{i}/{len(data_loader)}]  eta: {eta_string}  "
    log_msg += f"{str(metric_logger)}  mem: {memory:.0f}  max mem: {max_memory:.0f}"
    return log_msg