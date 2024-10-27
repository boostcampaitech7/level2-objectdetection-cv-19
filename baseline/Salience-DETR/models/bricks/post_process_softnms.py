import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
import torch
import torchvision
from torch import nn
from torchvision.ops import box_convert

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS implementation.
    Args:
        boxes (Tensor[N, 4]): boxes in [x1, y1, x2, y2] format.
        scores (Tensor[N]): scores for each box.
        iou_threshold (float): IoU threshold to decay scores.
        sigma (float): Variance for Gaussian decay.
        score_threshold (float): Minimum score to retain after decay.
    Returns:
        keep (Tensor): indexes of boxes to keep.
    """
    N = boxes.shape[0]
    for i in range(N):
        max_score_idx = scores[i:].argmax() + i
        if scores[max_score_idx] < score_threshold:
            break

        # Swap the highest score box to the current position
        boxes[i], boxes[max_score_idx] = boxes[max_score_idx].clone(), boxes[i].clone()
        scores[i], scores[max_score_idx] = scores[max_score_idx].clone(), scores[i].clone()

        # Compute IoU between the highest score box and the rest
        ious = box_ops.box_iou(boxes[i].unsqueeze(0), boxes[i + 1:])[0]

        # Apply Gaussian decay to scores based on IoU
        decay = torch.exp(-(ious ** 2) / sigma)
        scores[i + 1:] *= decay

    # Filter boxes by score threshold
    keep = scores > score_threshold
    return keep


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self,
        select_box_nums_for_evaluation=100,
        nms_iou_threshold=0.5,  # Soft-NMS IoU threshold 적용
        confidence_score=-1,
        sigma=0.5,  # Soft-NMS Gaussian decay parameter
    ):
        super().__init__()
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_score = confidence_score
        self.sigma = sigma  # Soft-NMS sigma value for decay

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # Sigmoid 변환
        prob = out_logits.sigmoid()

        # 상위 예측값 추출
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1),
            self.select_box_nums_for_evaluation,
            dim=1,
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="trunc")
        labels = topk_indexes % out_logits.shape[2]
        
        # bbox 변환
        boxes = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # 절대 좌표 변환
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 초기값 설정
        item_indice = [torch.ones_like(score, dtype=torch.bool) for score in scores]  # 기본적으로 모든 항목 유지

        # 신뢰도 필터링
        if self.confidence_score > 0:
            item_indice = [score > self.confidence_score for score in scores]

        # Soft-NMS 적용
        if self.nms_iou_threshold > 0:
            soft_nms_indice = [
                soft_nms(box, score, iou_threshold=self.nms_iou_threshold, sigma=self.sigma)
                for box, score in zip(boxes, scores)
            ]
            item_indice = [
                item_index & soft_nms_index
                for item_index, soft_nms_index in zip(item_indice, soft_nms_indice)
            ]

        # 필터된 결과만 남기기
        scores = [score[item_index] for score, item_index in zip(scores, item_indice)]
        boxes = [box[item_index] for box, item_index in zip(boxes, item_indice)]
        labels = [label[item_index] for label, item_index in zip(labels, item_indice)]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

class SegmentationPostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes, input_sizes, batched_input_size):
        out_logits, out_bbox, out_mask = (
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_masks"],
        )

        assert len(out_logits) == len(target_sizes)
        assert len(batched_input_size) == 2

        # we average queries of the same class to get onehot segmentation image
        out_class = out_logits.argmax(-1)
        num_class = out_logits.shape[-1]
        result_masks = []
        for image_id in range(len(out_logits)):
            result_masks_per_image = []
            for cur_class in range(num_class):
                class_index = out_class[image_id] == cur_class
                mask_per_class = out_mask[image_id][class_index].sigmoid()
                if mask_per_class.numel() == 0:
                    mask_per_class = mask_per_class.new_zeros((1, *mask_per_class.shape[-2:]))
                mask_per_class = mask_per_class.mean(0)
                result_masks_per_image.append(mask_per_class)
            result_masks_per_image = torch.stack(result_masks_per_image, 0)
            result_masks.append(result_masks_per_image)
        result_masks = torch.stack(result_masks, 0)

        # upsample masks with 1/4 resolution to input image shapes
        result_masks = F.interpolate(
            result_masks,
            size=batched_input_size,
            mode="bilinear",
            align_corners=False,
        )

        # resize masks to original shapes and transform onehot into class
        mask_results = []
        for mask, (height, width), (out_height, out_width) in zip(
            result_masks,
            input_sizes,
            target_sizes,
        ):
            mask = F.interpolate(
                mask[None, :, :height, :width],
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            mask_results.append({"masks": mask.argmax(0)})

        return mask_results
