import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor
from cityscapes_dataset import CityscapesFineDataset
from zero_shot_utils_box_prompts import CLASS_TO_ID, sample_boxes_from_gt

ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"
CHECKPOINT_PATH = r"C:\Users\dowen\Desktop\csci 490dpl final project\sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_MAP = {
    7: (128, 64, 128),
    8: (244, 35, 232),
    11: (70, 70, 70),
    24: (220, 20, 60),
    26: (0, 0, 142),
}


def build_sam_predictor(checkpoint_path, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def colorize_labels(label_map):
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in COLOR_MAP.items():
        mask = label_map == cid
        rgb[mask] = color
    return rgb


def draw_boxes_on_image(img_np, boxes, box_classes):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box, cid in zip(boxes, box_classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        color_rgb = COLOR_MAP.get(int(cid), (255, 255, 0))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def run_and_save_box_examples(num_examples=5):
    out_dir = "out_examples_box_prompts"
    os.makedirs(out_dir, exist_ok=True)
    ds = CityscapesFineDataset(ROOT, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    predictor = build_sam_predictor(CHECKPOINT_PATH, model_type="vit_h", device=DEVICE)
    count = 0
    idx_global = 0

    for _, gt_tensor, img_path in loader:
        img = Image.open(img_path[0]).convert("RGB")
        img_np = np.array(img)
        gt_np = gt_tensor[0].numpy()

        boxes, box_classes = sample_boxes_from_gt(gt_np)
        if boxes is None:
            idx_global += 1
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

        gt_color = colorize_labels(gt_np)
        pred_color = colorize_labels(pred)
        boxes_img = draw_boxes_on_image(img_np, boxes, box_classes)

        h_img, w_img, _ = img_np.shape
        collage = np.zeros((2 * h_img, 2 * w_img, 3), dtype=np.uint8)
        collage[0:h_img, 0:w_img] = img_np
        collage[0:h_img, w_img:2 * w_img] = gt_color
        collage[h_img:2 * h_img, 0:w_img] = boxes_img
        collage[h_img:2 * h_img, w_img:2 * w_img] = pred_color

        base = f"example_{idx_global:03d}_box_collage.png"
        Image.fromarray(collage).save(os.path.join(out_dir, base))

        count += 1
        idx_global += 1
        if count >= num_examples:
            break


if __name__ == "__main__":
    run_and_save_box_examples(num_examples=5)
