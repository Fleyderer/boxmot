# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import annotations
from collections import deque
from typing import Literal
from pathlib import Path

import lap
import numpy as np
import torch
from scipy.spatial.distance import cdist
from fastdist import fastdist

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.puretracker.storage import TrackState, TrackStorage, EmbeddingHandler
from boxmot.utils.iou import AssociationFunction
from boxmot.trackers.basetracker import BaseTracker


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

# TODO: maybe do it only once when creating


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


class PureTrackNew(BaseTracker):
    """
    PureTrackerNEW: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

    Args:
        track_thresh (float, optional): Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
        match_thresh (float, optional): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        track_buffer (int, optional): Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
        frame_rate (int, optional): Frame rate of the video being processed. Used to scale the track buffer size.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,

        track_high_thresh: float,
        track_low_thresh: float,
        track_new_thresh: float,
        match_thresh: float,

        dets_storage_size: int,
        tracks_storage_size: int,
        track_buffer: int,
        frame_rate: int = 30,
        per_class: bool = False,

        with_reid: bool = True,
        emb_mode: str = "ema",
        emb_ema_alpha: float = 0.9,
        emb_max_count: int = 10,

        iou_thresh: float = 0.5,
        emb_thresh: float = 0.5,
    ):
        super().__init__(per_class=per_class)
        self.tracks_storage = TrackStorage(size=tracks_storage_size,
                                           auto_increase=True)

        self.active_pool: np.ndarray = np.empty(0, dtype=int)
        self.lost_pool: np.ndarray = np.empty(0, dtype=int)
        self.removed_pool: np.ndarray = np.empty(0, dtype=int)

        EmbeddingHandler._mode = emb_mode
        EmbeddingHandler._ema_alpha = emb_ema_alpha
        EmbeddingHandler._max_len = emb_max_count

        self.frame_count = 0
        self.track_buffer = track_buffer
        self.per_class = per_class

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_new_thresh = track_new_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        self.with_reid = with_reid

        if self.with_reid:
            self.reid_model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half).model
        else:
            self.reid_model = None
        self.iou_thresh = iou_thresh
        self.emb_thresh = emb_thresh

    def _split_detections(self, dets: np.ndarray, embs: np.ndarray):
        dets = np.hstack((dets, np.arange(len(dets)).reshape(-1, 1)))
        dets_xywh = dets.copy()
        dets_xywh[:, :4] = xyxy2xywh(dets_xywh[:, :4])
        confs = dets[:, 4]
        inds_first = confs > self.track_high_thresh

        inds_higher_low = confs > self.track_low_thresh
        inds_lower_high = confs < self.track_high_thresh
        inds_second = np.logical_and(inds_higher_low, inds_lower_high)

        dets_high = dets[inds_first]
        dets_low = dets[inds_second]
        dets_high_xywh = dets_xywh[inds_first]
        dets_low_xywh = dets_xywh[inds_second]
        embs_high = embs[inds_first] if embs is not None else None

        return dets, dets_high, dets_low, dets_high_xywh, dets_low_xywh, embs_high

    def process_matches(self,
                        tracks_matches_pool: np.ndarray,
                        dets: np.ndarray, embs: np.ndarray = None):

        # Split tracks pool to update and reactivate
        update_tracks_pool = np.intersect1d(
            tracks_matches_pool,
            self.tracks_storage.states == TrackState.Tracked)

        reactivate_tracks_pool = np.intersect1d(
            tracks_matches_pool,
            self.tracks_storage.states != TrackState.Tracked)

        # Split dets pool to update and reactivate
        update_dets = dets[np.in1d(tracks_matches_pool,
                                   update_tracks_pool)]
        reactivate_dets = dets[np.in1d(tracks_matches_pool,
                                       reactivate_tracks_pool)]

        if embs is not None:
            update_embs = embs[np.in1d(tracks_matches_pool,
                                       update_tracks_pool)]
            reactivate_embs = embs[np.in1d(tracks_matches_pool,
                                           reactivate_tracks_pool)]
        else:
            update_embs = None
            reactivate_embs = None

        # Update, reactivate
        self.tracks_storage.update(update_tracks_pool, update_dets,
                                   self.frame_count, update_embs)

        self.tracks_storage.reactivate(reactivate_tracks_pool, reactivate_dets,
                                       self.frame_count, reactivate_embs)

        return update_tracks_pool, reactivate_tracks_pool

    def iou_distance(self, tracks_pool, dets):

        if len(tracks_pool) == 0 or len(dets) == 0:
            return np.zeros((len(tracks_pool),
                             len(dets)), dtype=np.float32)

        ious = AssociationFunction.iou_batch(
            xywh2xyxy(self.tracks_storage.means[tracks_pool][:, :4]),
            dets)

        return 1 - ious

    @staticmethod
    def pool_from_matches(tracks_pool: np.ndarray, dets_pool: np.ndarray,
                          matches: list[tuple[int, int]]):
        if len(matches) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        tracks_matches, dets_matches = zip(*matches)
        tracks_matches_pool = tracks_pool[list(tracks_matches)]
        dets_matches_pool = dets_pool[list(dets_matches)]
        return tracks_matches_pool, dets_matches_pool

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self,
               dets: np.ndarray,
               img: np.ndarray = None,
               embs: np.ndarray = None) -> np.ndarray:

        self.check_inputs(dets, img)

        self.frame_count += 1

        activated_pool = np.empty(0, dtype=int)
        reactivated_pool = np.empty(0, dtype=int)
        lost_pool = np.empty(0, dtype=int)
        removed_pool = np.empty(0, dtype=int)

        (dets, 
         dets_high, dets_low, 
         dets_high_xywh, dets_low_xywh,
         embs_high) = self._split_detections(dets, embs)

        # Extract appearance embeddings
        if self.with_reid and embs is None:
            embs_high = self.reid_model.get_features(dets_high[:, 0:4], img)
        else:
            embs_high = embs_high if embs_high is not None else None

        """ Add newly detected tracklets to tracked_stracks"""

        # active tracks, but not activated yet
        unconfirmed_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == False)

        # active and activated
        tracked_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == True)

        """ Step 2: First association, with high conf detection boxes"""
        tracks_pool = np.union1d(tracked_pool, self.lost_pool)
        dets_high_pool = np.arange(len(dets_high))

        # Predict the current location with KF
        self.tracks_storage.multi_predict(tracks_pool)

        iou_dists = self.iou_distance(tracks_pool, dets_high)

        iou_dists = fuse_score(
            iou_dists, dets_high[:, 4])

        if self.with_reid:
            emb_dists = embedding_distance(
                self.tracks_storage.embs[tracks_pool],
                embs_high) / 2.0

            emb_dists[emb_dists > self.emb_thresh] = 1.0
            emb_dists[iou_dists > self.iou_thresh] = 1.0
            dists = np.minimum(iou_dists, emb_dists)
        else:
            dists = iou_dists

        # Matches, unmatched tracks ids, unmatched detections ids
        matches, u_tracks, u_dets_high = linear_assignment(
            dists, thresh=self.match_thresh)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            tracks_pool, dets_high_pool, matches)

        unmatched_tracks_pool = tracks_pool[u_tracks]
        unmatched_dets_high_pool = dets_high_pool[list(u_dets_high)]

        # Update, reactivate
        high_activated_pool, high_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_high_xywh[dets_matches_pool], embs_high[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, high_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, high_reactivated_pool)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections

        remain_tracks_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states == TrackState.Tracked)

        dets_low_pool = np.arange(len(dets_low))

        iou_dists = self.iou_distance(remain_tracks_pool, dets_low)

        matches, u_tracks, u_dets_low = linear_assignment(
            iou_dists, thresh=0.5)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            remain_tracks_pool, dets_low_pool, matches)

        unmatched_tracks_pool = remain_tracks_pool[u_tracks]
        unmatched_dets_low_pool = u_dets_low

        # Update, reactivate
        low_activated_pool, low_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_low_xywh[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, low_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, low_reactivated_pool)

        lost_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states != TrackState.Lost)

        self.tracks_storage.states[lost_pool] = TrackState.Lost

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Unmatched detections from first association (high conf dets)

        iou_dists = self.iou_distance(
            unconfirmed_pool, dets_high[unmatched_dets_high_pool])

        iou_dists = fuse_score(
            iou_dists, dets_high[unmatched_dets_high_pool, 4])

        matches, u_tracks, u_dets_high = linear_assignment(
            iou_dists, thresh=0.7)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            unconfirmed_pool, unmatched_dets_high_pool, matches)

        unmatched_tracks_pool = unconfirmed_pool[u_tracks]
        unmatched_remain_high_dets_pool = unmatched_dets_high_pool[u_dets_high]

        self.tracks_storage.update(tracks_matches_pool,
                                   dets_high_xywh[dets_matches_pool],
                                   self.frame_count, embs[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, tracks_matches_pool)

        self.tracks_storage.states[unmatched_tracks_pool] = TrackState.Removed

        removed_pool = np.union1d(removed_pool, unmatched_tracks_pool)

        """ Step 4: Init new stracks"""

        dets_to_tracks_pool = np.intersect1d(
            unmatched_remain_high_dets_pool,
            np.where(dets_high[:, 4] >= self.track_new_thresh)[0])

        new_tracks_pool = self.tracks_storage.activate(
            dets_high_xywh[dets_to_tracks_pool], self.frame_count,
            embs_high[dets_to_tracks_pool])

        activated_pool = np.union1d(activated_pool, new_tracks_pool)

        """ Step 5: Update state"""

        remove_filter = (
            self.frame_count - self.tracks_storage.frame_ids[self.lost_pool] > self.max_time_lost)
        remove_pool = self.lost_pool[remove_filter]

        self.tracks_storage.states[remove_pool] = TrackState.Removed
        removed_pool = np.union1d(removed_pool, remove_pool)

        self.active_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.states == TrackState.Tracked)

        self.active_pool = np.union1d(self.active_pool, activated_pool)
        self.active_pool = np.union1d(self.active_pool, reactivated_pool)

        self.lost_pool = np.setdiff1d(self.lost_pool, self.active_pool)
        self.lost_pool = np.union1d(self.lost_pool, lost_pool)
        self.lost_pool = np.setdiff1d(self.lost_pool, self.removed_pool)

        self.removed_pool = np.union1d(self.removed_pool, removed_pool)

        self.active_pool, self.lost_pool = self.remove_duplicates(
            self.active_pool, self.lost_pool
        )

        # Clean up removed tracks
        if self.frame_count % self.buffer_size == 0:
            save_pool = np.union1d(self.active_pool, self.lost_pool)
            self.tracks_storage.cleanup(save_pool)
            self.removed_pool = np.intersect1d(self.removed_pool, save_pool)

        output_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == True)

        xyxys = xywh2xyxy(self.tracks_storage.means[output_pool][:, :4])
        ids = output_pool.reshape(-1, 1)
        confs = self.tracks_storage.confs[output_pool].reshape(-1, 1)
        classes = self.tracks_storage.classes[output_pool].reshape(-1, 1)
        det_ids = self.tracks_storage.det_ids[output_pool].reshape(-1, 1)

        outputs = np.hstack((xyxys, ids, confs, classes, det_ids))

        return outputs

    def remove_duplicates(self, tracksa_pool, tracksb_pool):
        pdist = 1 - AssociationFunction.iou_batch(
            xywh2xyxy(self.tracks_storage.means[tracksa_pool][:, :4]),
            xywh2xyxy(self.tracks_storage.means[tracksb_pool][:, :4]),
        )

        tracksa_dup, tracksb_dup = np.where(pdist < 0.15)

        tracksa_dup_pool = tracksa_pool[tracksa_dup]
        tracksb_dup_pool = tracksb_pool[tracksb_dup]

        timesa = self.tracks_storage.frame_ids[tracksa_dup_pool] - \
            self.tracks_storage.start_frames[tracksa_dup_pool]
        timesb = self.tracks_storage.frame_ids[tracksb_dup_pool] - \
            self.tracks_storage.start_frames[tracksb_dup_pool]

        # Continue from here --->

        finala_pool = np.setdiff1d(tracksa_pool,
                                   tracksa_dup_pool[timesa < timesb])
        finalb_pool = np.setdiff1d(tracksb_pool,
                                   tracksb_dup_pool[timesb < timesa])

        return finala_pool, finalb_pool
