import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from segment_anything import sam_model_registry
from cityscapes_dataset import CityscapesFineDataset
from zero_shot_utils_box_prompts import (
    CLASS_TO_ID,
    init_iou_accumulators,
    update_iou_accumulators,
    compute_iou_from_accumulators,
)

ROOT = r"C:\Users\dowen\Desktop\csci 490dpl final project"
CHECKPOINT_PATH = r"C:\Users\dowen\Desktop\csci 490dpl final project\sam_vit_h_4b8939.pth"
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
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-3
MAX_TRAIN_SAMPLES = 80000
MAX_VAL_SAMPLES = 20000


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


def preprocess_image_batch(img_batch):
    img_batch = img_batch.to(DEVICE)
    img_batch = F.interpolate(img_batch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1, 3, 1, 1) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(1, 3, 1, 1) / 255.0
    img_batch = (img_batch - mean) / std
    return img_batch


def preprocess_labels_batch(gt_batch):
    gt_batch = gt_batch.to(DEVICE)
    gt_batch = F.interpolate(gt_batch.unsqueeze(1).float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").squeeze(1).long()
    mapped = torch.full_like(gt_batch, BACKGROUND_TRAIN_ID)
    for cs_id, train_id in CITYSCAPES_TO_TRAIN.items():
        mapped[gt_batch == cs_id] = train_id
    mapped[gt_batch == 255] = 255
    return mapped


def logits_to_cityscapes_ids(logits, out_size):
    preds = torch.argmax(logits, dim=1)
    preds = F.interpolate(preds.unsqueeze(1).float(), size=out_size, mode="nearest").squeeze(1).long()
    cs_pred = torch.full_like(preds, 255)
    for train_id, cs_id in TRAIN_TO_CITYSCAPES.items():
        cs_pred[preds == train_id] = cs_id
    return cs_pred


def compute_val_miou(sam, head, val_loader):
    sam.eval()
    head.eval()
    inter, union = init_iou_accumulators()
    with torch.no_grad():
        for imgs, gts, _ in val_loader:
            imgs = imgs.to(DEVICE)
            gts = gts.to(DEVICE)
            h0, w0 = gts.shape[-2], gts.shape[-1]
            imgs_norm = preprocess_image_batch(imgs)
            feats = sam.image_encoder(imgs_norm)
            logits = head(feats, out_size=(IMAGE_SIZE, IMAGE_SIZE))
            cs_pred = logits_to_cityscapes_ids(logits, out_size=(h0, w0))
            for i in range(gts.shape[0]):
                gt_i = gts[i].cpu().numpy()
                pred_i = cs_pred[i].cpu().numpy()
                update_iou_accumulators(gt_i, pred_i, inter, union)
    class_ious, miou = compute_iou_from_accumulators(inter, union)
    return class_ious, miou


def main():
    full_train_ds = CityscapesFineDataset(ROOT, split="train")
    full_val_ds = CityscapesFineDataset(ROOT, split="val")

    train_indices = list(range(len(full_train_ds)))[:MAX_TRAIN_SAMPLES]
    val_indices = list(range(len(full_val_ds)))[:MAX_VAL_SAMPLES]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sam = build_frozen_sam(CHECKPOINT_PATH, model_type="vit_h", device=DEVICE)
    head = SamCityscapesHead(in_channels=256, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(1, EPOCHS + 1):
        sam.eval()
        head.train()
        running_loss = 0.0
        num_pixels = 0
        num_steps = len(train_loader)

        for step, (imgs, gts, _) in enumerate(train_loader, start=1):
            imgs = imgs.to(DEVICE)
            gts = gts.to(DEVICE)

            imgs_norm = preprocess_image_batch(imgs)
            labels = preprocess_labels_batch(gts)

            with torch.no_grad():
                feats = sam.image_encoder(imgs_norm)

            logits = head(feats, out_size=(IMAGE_SIZE, IMAGE_SIZE))
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid = (labels != 255).sum().item()
            running_loss += loss.item() * valid
            num_pixels += valid

            if step % 50 == 0 or step == num_steps:
                avg_step_loss = running_loss / max(1, num_pixels)
                print(f"Epoch {epoch} Step {step}/{num_steps} avg_loss_per_pixel {avg_step_loss:.6f}")

        avg_loss = running_loss / max(1, num_pixels)
        class_ious, miou = compute_val_miou(sam, head, val_loader)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train loss per labeled pixel: {avg_loss:.6f}")
        print("  Val per-class IoU:")
        for cid, iou in class_ious.items():
            print(f"    class {cid}: {iou:.4f}")
        print(f"  Val mIoU: {miou:.4f}")

    os.makedirs("out_models_cityscapes_head", exist_ok=True)
    torch.save(head.state_dict(), os.path.join("out_models_cityscapes_head", "sam_cityscapes_head.pth"))
    print("Saved head to out_models_cityscapes_head/sam_cityscapes_head.pth")


if __name__ == "__main__":
    main()
