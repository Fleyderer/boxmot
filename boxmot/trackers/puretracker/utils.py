import pathlib
import shutil
import configparser

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


def xywh2xyxy_clip(x: np.ndarray | torch.Tensor, img_size: tuple[int, int]):
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

    # Clip coordinates to image size
    y[..., 0] = np.clip(y[..., 0], 0, img_size[1])
    y[..., 1] = np.clip(y[..., 1], 0, img_size[0])
    y[..., 2] = np.clip(y[..., 2], 0, img_size[1])
    y[..., 3] = np.clip(y[..., 3], 0, img_size[0])
    
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
    bboxes1_x1, bboxes1_y1 = bboxes1[..., 0] - \
        half_w1, bboxes1[..., 1] - half_h1
    bboxes1_x2, bboxes1_y2 = bboxes1[..., 0] + \
        half_w1, bboxes1[..., 1] + half_h1

    bboxes2_x1, bboxes2_y1 = bboxes2[..., 0] - \
        half_w2, bboxes2[..., 1] - half_h2
    bboxes2_x2, bboxes2_y2 = bboxes2[..., 0] + \
        half_w2, bboxes2[..., 1] + half_h2

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


def iou_vr_batch_xywh(bboxes1, bboxes2) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Intersection over Union (IoU) and Visibility Ratio (VR) 
    of two sets of bounding boxes.


    Args:
        bboxes1 (np.ndarray): 
            An array of bounding boxes in (x, y, w, h) format.
            We assume that bboxes1 are tracks.
        bboxes2 (np.ndarray): 
            An array of bounding boxes in (x, y, w, h) format.
            We assume that bboxes2 are detections.

    Returns:
        np.ndarray: 
            An array of IoU values for each 
            pair of bounding boxes in bboxes1 and bboxes2.
        np.ndarray: 
            An array of VR values for bboxes2 (detections).
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Calculate half widths and half heights
    half_w1, half_h1 = bboxes1[..., 2] / 2, bboxes1[..., 3] / 2
    half_w2, half_h2 = bboxes2[..., 2] / 2, bboxes2[..., 3] / 2

    # Calculate the coordinates of the top-left and bottom-right corners
    bboxes1_x1, bboxes1_y1 = bboxes1[..., 0] - \
        half_w1, bboxes1[..., 1] - half_h1
    bboxes1_x2, bboxes1_y2 = bboxes1[..., 0] + \
        half_w1, bboxes1[..., 1] + half_h1

    bboxes2_x1, bboxes2_y1 = bboxes2[..., 0] - \
        half_w2, bboxes2[..., 1] - half_h2
    bboxes2_x2, bboxes2_y2 = bboxes2[..., 0] + \
        half_w2, bboxes2[..., 1] + half_h2

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

    # Compute Visibility Ratio (VR) matrix
    # Assuming bboxes1 represents occluders for bboxes2,
    # the visible area is the object's area minus the intersection area.
    vr = (bboxes2_area - inter_area) / bboxes2_area

    # Ensure values are within [0, 1]
    # visibility_ratio = np.clip(visibility_ratio, 0.0, 1.0)

    return iou, vr


def aiou_batch_xywh(bboxes1, bboxes2) -> tuple[np.ndarray, np.ndarray]:

    # Compute aspect ratios
    bboxes1_aspect_ratio = bboxes1[:, 2] / bboxes1[:, 3]
    bboxes2_aspect_ratio = bboxes2[:, 2] / bboxes2[:, 3]

    # Expand dimensions
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Calculate half widths and half heights
    half_w1, half_h1 = bboxes1[..., 2] / 2, bboxes1[..., 3] / 2
    half_w2, half_h2 = bboxes2[..., 2] / 2, bboxes2[..., 3] / 2

    # Calculate the coordinates of the top-left and bottom-right corners
    bboxes1_x1, bboxes1_y1 = bboxes1[..., 0] - \
        half_w1, bboxes1[..., 1] - half_h1
    bboxes1_x2, bboxes1_y2 = bboxes1[..., 0] + \
        half_w1, bboxes1[..., 1] + half_h1

    bboxes2_x1, bboxes2_y1 = bboxes2[..., 0] - \
        half_w2, bboxes2[..., 1] - half_h2
    bboxes2_x2, bboxes2_y2 = bboxes2[..., 0] + \
        half_w2, bboxes2[..., 1] + half_h2

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
    ious = inter_area / union_area

    # Compute alpha
    arctan_diff = (np.arctan(bboxes1_aspect_ratio[:, None]) -
                   np.arctan(bboxes2_aspect_ratio[None, :]))
    v = 1 - ((4 / (np.pi ** 2)) * arctan_diff ** 2)
    alphas = v / (1 - ious + v)

    return ious, alphas


def camera_update(boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Update coordinates of a batch of boxes using a camera transformation
    
    Args:
        boxes: Array of boxes in [x_center, y_center, width, height] 
               format with shape [B, 4]
        transform: Transformation matrix with shape [3, 3]
    
    Returns:
        Updated boxes in [x_center, y_center, width, height] 
        format with shape [B, 4]
    """
    batch_size = boxes.shape[0]
    
    # Extract components
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Calculate corners for each box
    half_w = w / 2
    half_h = h / 2
    
    # Calculate all corners for all boxes at once
    # For each box, calculate its 4 corners
    x1 = cx - half_w  # top-left x
    y1 = cy - half_h  # top-left y
    x2 = cx + half_w  # bottom-right x
    y2 = cy + half_h  # bottom-right y
    
    # Create all corners as homogeneous coordinates [batch_size, 4, 3]
    corners = np.zeros((batch_size, 4, 3))
    
    # Populate corners: top-left, top-right, bottom-right, bottom-left
    corners[:, 0, :] = np.column_stack([x1, y1, np.ones(batch_size)])  # top-left
    corners[:, 1, :] = np.column_stack([x2, y1, np.ones(batch_size)])  # top-right
    corners[:, 2, :] = np.column_stack([x2, y2, np.ones(batch_size)])  # bottom-right
    corners[:, 3, :] = np.column_stack([x1, y2, np.ones(batch_size)])  # bottom-left
    
    # Reshape to apply transform to all corners at once
    corners_flat = corners.reshape(-1, 3)  # [B*4, 3]
    
    # Apply transform
    corners_trans_flat = corners_flat @ transform.T  # [B*4, 3]
    
    # Reshape back
    corners_trans = corners_trans_flat.reshape(batch_size, 4, 3)  # [B, 4, 3]
    
    # Extract transformed coordinates
    trans_x = corners_trans[:, :, 0]  # [B, 4]
    trans_y = corners_trans[:, :, 1]  # [B, 4]
    
    # Calculate new bounding box dimensions
    new_x1 = np.min(trans_x, axis=1)
    new_y1 = np.min(trans_y, axis=1)
    new_x2 = np.max(trans_x, axis=1)
    new_y2 = np.max(trans_y, axis=1)
    
    # Compute new centers and dimensions
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    new_cx = new_x1 + new_w / 2
    new_cy = new_y1 + new_h / 2
    
    # Return in [x_center, y_center, width, height] format
    return np.stack([new_cx, new_cy, new_w, new_h], axis=1)