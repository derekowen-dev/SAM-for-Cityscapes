import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor
from cityscapes_dataset import CityscapesFineDataset
from zero_shot_utils_box_prompts import (
    sample_boxes_from_gt,
    init_iou_accumulators,
    update_iou_accumulators,
    compute_iou_from_accumulators,
)

ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"
CHECKPOINT_PATH = r"C:\Users\dowen\Desktop\csci 490dpl final project\sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_sam_predictor(checkpoint_path, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def run_box_zero_shot():
    ds = CityscapesFineDataset(ROOT, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    predictor = build_sam_predictor(CHECKPOINT_PATH, model_type="vit_h", device=DEVICE)
    inter, union = init_iou_accumulators()

    for idx, (_, gt_tensor, img_path) in enumerate(loader):
        img = Image.open(img_path[0]).convert("RGB")
        img_np = np.array(img)
        gt_np = gt_tensor[0].numpy()

        boxes, box_classes = sample_boxes_from_gt(gt_np)
        if boxes is None:
            continue

        predictor.set_image(img_np)
        h, w = gt_np.shape
        pred = np.full((h, w), 255, dtype=np.int64)

        for box, cid in zip(boxes, box_classes):
            b = np.array(box, dtype=np.float32)[None, :]
            masks, scores, _ = predictor.predict(
                box=b,
                multimask_output=True,
            )
            k = int(np.argmax(scores))
            mask = masks[k].astype(bool)
            if mask.shape != pred.shape:
                if mask.shape[::-1] == pred.shape:
                    mask = mask.T
                else:
                    continue
            pred[mask] = int(cid)

        update_iou_accumulators(gt_np, pred, inter, union)

        if (idx + 1) % 50 == 0:
            class_ious, miou = compute_iou_from_accumulators(inter, union)
            print(f"Processed {idx+1} images, current mIoU: {miou:.4f}")

    class_ious, miou = compute_iou_from_accumulators(inter, union)
    print("Final per-class IoU (box prompts):")
    for cid, iou in class_ious.items():
        print(f"  class {cid}: {iou:.4f}")
    print(f"Final mIoU (box prompts): {miou:.4f}")


if __name__ == "__main__":
    run_box_zero_shot()
