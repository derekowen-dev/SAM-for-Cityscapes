import numpy as np
import cv2

CLASS_TO_ID = {
    "road": 7,
    "sidewalk": 8,
    "building": 11,
    "person": 24,
    "car": 26,
}

STUFF_CLASSES = {7, 8, 11}


def farthest_point_indices(coords, k):
    n = coords.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)
    idxs = np.empty(k, dtype=int)
    idxs[0] = np.random.randint(n)
    dist = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        current = coords[idxs[i - 1]]
        d = np.sum((coords - current) ** 2, axis=1)
        dist = np.minimum(dist, d)
        idxs[i] = int(np.argmax(dist))
    return idxs


def sample_points_from_gt(gt_label_map, classes=CLASS_TO_ID):
    H, W = gt_label_map.shape
    point_coords = []
    point_labels = []
    point_classes = []
    kernel = np.ones((5, 5), np.uint8)
    for cname, cid in classes.items():
        mask_full = (gt_label_map == cid).astype(np.uint8)
        if mask_full.sum() == 0:
            continue
        mask_eroded = cv2.erode(mask_full, kernel, iterations=1)
        if mask_eroded.sum() > 0:
            ys, xs = np.where(mask_eroded == 1)
        else:
            ys, xs = np.where(mask_full == 1)
        if len(xs) == 0:
            continue
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        if cid in STUFF_CLASSES:
            n = min(3, len(xs))
            idxs = farthest_point_indices(coords, n)
        else:
            if cid == CLASS_TO_ID["car"]:
                pixels_per_point = 10000
                max_points = 10
            elif cid == CLASS_TO_ID["person"]:
                pixels_per_point = 4000
                max_points = 8
            else:
                pixels_per_point = 10000
                max_points = 10
            est = max(1, len(xs) // pixels_per_point)
            n = min(max_points, est)
            n = min(n, len(xs))
            if n <= 0:
                n = 1
            idxs = farthest_point_indices(coords, n)
        for i in idxs:
            x, y = int(coords[i, 0]), int(coords[i, 1])
            point_coords.append([x, y])
            point_labels.append(1)
            point_classes.append(cid)
    if len(point_coords) == 0:
        return None, None, None
    return np.array(point_coords), np.array(point_labels), np.array(point_classes)


def sample_boxes_from_gt(gt_label_map, classes=CLASS_TO_ID):
    H, W = gt_label_map.shape
    all_boxes = []
    all_classes = []
    for cname, cid in classes.items():
        mask_full = (gt_label_map == cid).astype(np.uint8)
        if mask_full.sum() == 0:
            continue
        num_labels, labels = cv2.connectedComponents(mask_full)
        boxes = []
        centers = []
        if cid in STUFF_CLASSES:
            min_pixels = 1000
            max_boxes = 5
        else:
            min_pixels = 50
            if cid == CLASS_TO_ID["car"]:
                max_boxes = 60
            elif cid == CLASS_TO_ID["person"]:
                max_boxes = 80
            else:
                max_boxes = 40
        for lbl in range(1, num_labels):
            comp = labels == lbl
            area = int(comp.sum())
            if area < min_pixels:
                continue
            ys, xs = np.where(comp)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            boxes.append([x1, y1, x2, y2])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            centers.append([cx, cy])
        if len(boxes) == 0:
            continue
        centers = np.array(centers, dtype=np.float32)
        if len(boxes) > max_boxes:
            idxs = farthest_point_indices(centers, max_boxes)
        else:
            idxs = np.arange(len(boxes), dtype=int)
        for i in idxs:
            all_boxes.append(boxes[i])
            all_classes.append(cid)
    if len(all_boxes) == 0:
        return None, None
    return np.array(all_boxes, dtype=np.float32), np.array(all_classes, dtype=np.int64)


def init_iou_accumulators():
    inter = {cid: 0 for cid in CLASS_TO_ID.values()}
    union = {cid: 0 for cid in CLASS_TO_ID.values()}
    return inter, union


def update_iou_accumulators(gt, pred, inter, union):
    for cid in CLASS_TO_ID.values():
        gt_c = gt == cid
        pred_c = pred == cid
        i = (gt_c & pred_c).sum()
        u = (gt_c | pred_c).sum()
        inter[cid] += int(i)
        union[cid] += int(u)


def compute_iou_from_accumulators(inter, union):
    class_ious = {}
    for cid in inter.keys():
        if union[cid] > 0:
            class_ious[cid] = inter[cid] / union[cid]
        else:
            class_ious[cid] = float("nan")
    valid = [v for v in class_ious.values() if not np.isnan(v)]
    miou = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    return class_ious, miou
