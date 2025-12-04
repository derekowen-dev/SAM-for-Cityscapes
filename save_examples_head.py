import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry, SamPredictor
from cityscapes_dataset import CityscapesFineDataset
from zero_shot_utils_box_prompts import CLASS_TO_ID, sample_boxes_from_gt

ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"
CHECKPOINT_PATH = r"C:\Users\dowen\Desktop\csci 490dpl final project\sam_vit_h_4b8939.pth"
HEAD_PATH = r"C:\Users\dowen\Desktop\csci 490dpl final project\out_models_cityscapes_head\sam_cityscapes_head.pth"
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


def build_frozen_sam(checkpoint_path, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False
    return sam


def build_sam_predictor(checkpoint_path, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


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


def run_and_save_head_examples(num_examples=5):
    out_dir = "out_examples_head"
    os.makedirs(out_dir, exist_ok=True)

    val_ds = CityscapesFineDataset(ROOT, split="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    sam_backbone = build_frozen_sam(CHECKPOINT_PATH, model_type="vit_h", device=DEVICE)
    head = SamCityscapesHead(in_channels=256, num_classes=NUM_CLASSES).to(DEVICE)
    state = torch.load(HEAD_PATH, map_location=DEVICE)
    head.load_state_dict(state)
    head.eval()

    predictor_box = build_sam_predictor(CHECKPOINT_PATH, model_type="vit_h", device=DEVICE)

    count = 0
    idx_global = 0

    for imgs, gts, img_paths in val_loader:
        img_tensor = imgs.to(DEVICE)
        gt_tensor = gts.to(DEVICE)
        img = Image.open(img_paths[0]).convert("RGB")
        img_np = np.array(img)
        gt_np = gt_tensor[0].cpu().numpy()
        h0, w0 = gt_np.shape

        imgs_norm = preprocess_image_batch(img_tensor)
        with torch.no_grad():
            feats = sam_backbone.image_encoder(imgs_norm)
            logits = head(feats, out_size=(IMAGE_SIZE, IMAGE_SIZE))
            cs_pred_head = logits_to_cityscapes_ids(logits, out_size=(h0, w0))
        pred_head_np = cs_pred_head[0].cpu().numpy()

        boxes, box_classes = sample_boxes_from_gt(gt_np)
        predictor_box.set_image(img_np)
        pred_box = np.full((h0, w0), 255, dtype=np.int64)
        if boxes is not None:
            for box, cid in zip(boxes, box_classes):
                b = np.array(box, dtype=np.float32)[None, :]
                masks, scores, _ = predictor_box.predict(
                    box=b,
                    multimask_output=True,
                )
                k = int(np.argmax(scores))
                mask = masks[k].astype(bool)
                if mask.shape != pred_box.shape:
                    if mask.shape[::-1] == pred_box.shape:
                        mask = mask.T
                    else:
                        continue
                pred_box[mask] = int(cid)

        gt_color = colorize_labels(gt_np)
        pred_box_color = colorize_labels(pred_box)
        pred_head_color = colorize_labels(pred_head_np)

        h_img, w_img, _ = img_np.shape
        collage = np.zeros((2 * h_img, 2 * w_img, 3), dtype=np.uint8)
        collage[0:h_img, 0:w_img] = img_np
        collage[0:h_img, w_img:2 * w_img] = gt_color
        collage[h_img:2 * h_img, 0:w_img] = pred_box_color
        collage[h_img:2 * h_img, w_img:2 * w_img] = pred_head_color

        base = f"example_{idx_global:03d}_head_collage.png"
        Image.fromarray(collage).save(os.path.join(out_dir, base))

        count += 1
        idx_global += 1
        if count >= num_examples:
            break


if __name__ == "__main__":
    run_and_save_head_examples(num_examples=5)
