import lap
import numpy as np
import torch
from scipy.spatial.distance import cdist


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def xywh2xyxy(x: np.ndarray | torch.Tensor):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    if x.shape[0] == 0:
        return x

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    if x.shape[0] == 0:
        return x

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def embedding_distance(tracks_embs: np.ndarray, dets_embs: np.ndarray):
    cost_matrix = np.zeros(
        (len(tracks_embs), len(dets_embs)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(
        0.0, cdist(tracks_embs, dets_embs, metric="cosine")
    )  # Nomalized features

    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, confs: np.ndarray):
    if cost_matrix.size == 0:
        return cost_matrix

    confs = np.expand_dims(confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    iou_sim = 1 - cost_matrix
    fuse_sim = iou_sim * confs
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def iou_batch_xywh(bboxes1, bboxes2) -> np.ndarray:
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Calculate half widths and half heights
    half_w1, half_h1 = bboxes1[..., 2] / 2, bboxes1[..., 3] / 2
    half_w2, half_h2 = bboxes2[..., 2] / 2, bboxes2[..., 3] / 2

    # Calculate the coordinates of the top-left and bottom-right corners
    bboxes1_x1, bboxes1_y1 = bboxes1[..., 0] - half_w1, bboxes1[..., 1] - half_h1
    bboxes1_x2, bboxes1_y2 = bboxes1[..., 0] + half_w1, bboxes1[..., 1] + half_h1

    bboxes2_x1, bboxes2_y1 = bboxes2[..., 0] - half_w2, bboxes2[..., 1] - half_h2
    bboxes2_x2, bboxes2_y2 = bboxes2[..., 0] + half_w2, bboxes2[..., 1] + half_h2

    # Determine the coordinates of the intersection rectangle
    xx1 = np.maximum(bboxes1_x1, bboxes2_x1)
    yy1 = np.maximum(bboxes1_y1, bboxes2_y1)
    xx2 = np.minimum(bboxes1_x2, bboxes2_x2)
    yy2 = np.minimum(bboxes1_y2, bboxes2_y2)

    # Compute the width and height of the intersection rectangle
    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)

    # Compute the area of the intersection rectangle
    inter_area = inter_w * inter_h

    # Compute the area of both the prediction and ground-truth rectangles
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    # Compute the intersection over union
    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = inter_area / union_area

    return iou