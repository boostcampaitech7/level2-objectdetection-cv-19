import PIL.Image
import numpy as np
import torch
from pycocotools import mask as coco_mask


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image_target_tuple):
        image, target = image_target_tuple
        if isinstance(image, (torch.Tensor, np.ndarray)):
            assert len(image.shape) == 3, "only one image is accepted"
            assert image.shape[-3] in [1, 3], "channels of images must be 1 or 3"
            _, h, w = image.shape
        elif isinstance(image, PIL.Image.Image):
            w, h = image.size
        else:
            raise TypeError(
                f"Now only torch.Tensor, PIL.Image.Image and np.ndarray "
                f"of an image is accepted but got type {type(image)}"
            )

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.as_tensor(classes, dtype=torch.int64)

        masks = None
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # adapt to result_file
        scores = None
        if anno and "score" in anno[0]:
            scores = [obj["score"] for obj in anno]
            scores = torch.as_tensor(scores, dtype=torch.float32)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if scores is not None:
            scores = scores[keep]

        target = {"boxes": boxes, "labels": classes, "image_id": target["image_id"]}
        if masks is not None:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints
        if scores is not None:
            target["scores"] = scores

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno], dtype=torch.float32)
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=torch.long
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
