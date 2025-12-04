import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from cityscapes_dataset import CityscapesFineDataset
from zero_shot_utils_box_prompts import sample_boxes_from_gt

PROJECT_ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "sam_vit_h_4b8939.pth")
HEAD_PATH = os.path.join(PROJECT_ROOT, "out_models_cityscapes_head", "sam_cityscapes_head.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CITYSCAPES_TO_TRAIN = {
    7: 0,
    8: 1,
    11: 2,
    24: 3,
    26: 4,
}
TRAIN_TO_CITYSCAPES = {v: k for k, v in CITYSCAPES_TO_TRAIN.items()}
BACKGROUND_TRAIN_ID = 5
NUM_CLASSES = 6
IMAGE_SIZE = 1024

COLOR_MAP = {
    7: (128, 64, 128),
    8: (244, 35, 232),
    11: (70, 70, 70),
    24: (220, 20, 60),
    26: (0, 0, 142),
}


class SamCityscapesHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv_out = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, out_size):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.conv_out(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


def preprocess_image_batch(img_batch):
    img_batch = img_batch.to(DEVICE)
    img_batch = F.interpolate(img_batch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1, 3, 1, 1) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(1, 3, 1, 1) / 255.0
    img_batch = (img_batch - mean) / std
    return img_batch


def logits_to_cityscapes_ids(logits, out_size):
    preds = torch.argmax(logits, dim=1)
    preds = F.interpolate(preds.unsqueeze(1).float(), size=out_size, mode="nearest").squeeze(1).long()
    cs_pred = torch.full_like(preds, 255)
    for train_id, cs_id in TRAIN_TO_CITYSCAPES.items():
        cs_pred[preds == train_id] = cs_id
    return cs_pred


def colorize_labels(label_map):
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in COLOR_MAP.items():
        mask = label_map == cid
        rgb[mask] = color
    return rgb


def main():
    seed_str = input("Enter random seed: ").strip()
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    out_dir = os.path.join(PROJECT_ROOT, "out_examples_main")
    os.makedirs(out_dir, exist_ok=True)

    val_ds = CityscapesFineDataset(PROJECT_ROOT, split="val")
    n = len(val_ds)
    indices = random.sample(range(n), 5)

    sam_model = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
    sam_model.to(device=DEVICE)
    sam_model.eval()

    predictor_box = SamPredictor(sam_model)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    head = SamCityscapesHead(in_channels=256, num_classes=NUM_CLASSES).to(DEVICE)
    state = torch.load(HEAD_PATH, map_location=DEVICE)
    head.load_state_dict(state)
    head.eval()

    for idx in indices:
        img_tensor, gt_tensor, img_paths = val_ds[idx]
        if isinstance(img_paths, (list, tuple)):
            img_path = img_paths[0]
        else:
            img_path = img_paths

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        gt_np = gt_tensor.numpy()
        h0, w0 = gt_np.shape

        gt_color = colorize_labels(gt_np)

        boxes, box_classes = sample_boxes_from_gt(gt_np)
        box_overlay = img_np.copy()
        pred_box = np.full((h0, w0), 255, dtype=np.int64)
        if boxes is not None:
            predictor_box.set_image(img_np)
            for box, cid in zip(boxes, box_classes):
                x1, y1, x2, y2 = map(int, box)
                color = COLOR_MAP.get(int(cid), (255, 255, 255))
                cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, 2)
                b = np.array(box, dtype=np.float32)[None, :]
                masks, scores, _ = predictor_box.predict(box=b, multimask_output=True)
                k = int(np.argmax(scores))
                mask = masks[k].astype(bool)
                if mask.shape != pred_box.shape:
                    if mask.shape[::-1] == pred_box.shape:
                        mask = mask.T
                    else:
                        continue
                pred_box[mask] = int(cid)
        box_seg_color = colorize_labels(pred_box)

        masks_auto = mask_generator.generate(img_np)
        sam_auto = img_np.copy()
        for m in masks_auto:
            seg = m["segmentation"]
            color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
            sam_auto[seg] = (0.5 * sam_auto[seg] + 0.5 * color).astype(np.uint8)

        img_batch = img_tensor.unsqueeze(0)
        imgs_norm = preprocess_image_batch(img_batch)
        with torch.no_grad():
            feats = sam_model.image_encoder(imgs_norm)
            logits = head(feats, out_size=(IMAGE_SIZE, IMAGE_SIZE))
            cs_pred_head = logits_to_cityscapes_ids(logits, out_size=(h0, w0))
        pred_head_np = cs_pred_head[0].cpu().numpy()
        head_seg_color = colorize_labels(pred_head_np)

        collage = np.zeros((2 * h0, 3 * w0, 3), dtype=np.uint8)
        collage[0:h0, 0:w0] = img_np
        collage[0:h0, w0:2 * w0] = gt_color
        collage[0:h0, 2 * w0:3 * w0] = box_overlay
        collage[h0:2 * h0, 0:w0] = box_seg_color
        collage[h0:2 * h0, w0:2 * w0] = sam_auto
        collage[h0:2 * h0, 2 * w0:3 * w0] = head_seg_color

        out_name = f"example_{idx:03d}_main.png"
        Image.fromarray(collage).save(os.path.join(out_dir, out_name))


if __name__ == "__main__":
    main()
