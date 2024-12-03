# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import annotations
from collections import deque
from typing import Literal
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from fastdist import fastdist

from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.puretracker.basetrack import (BaseTrack, TrackState,
                                                   EmbeddingHandler,
                                                   ClassStorage)
from boxmot.trackers.puretracker.storage import TrackStorage
from boxmot.utils.iou import AssociationFunction
from boxmot.utils.matching import iou_distance, linear_assignment
from boxmot.utils.ops import xyxy2xywh, xywh2xyxy
from boxmot.trackers.basetracker import BaseTracker


# def embedding_distance(tracks: list[STrack],
#                        detections: list[STrack],
#                        metric='cosine'):
#     """
#     :param tracks: list[STrack]
#     :param detections: list[STrack]
#     :param metric: str
#     :return: cost_matrix np.ndarray
#     """
#     cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
#     if cost_matrix.size == 0:
#         return cost_matrix
#     det_features = np.asarray(
#         [det.emb.get_last() for det in detections], dtype=np.float32
#     )
#     track_features = np.asarray(
#         [track.emb.get() for track in tracks], dtype=np.float32
#     )
#     cost_matrix = np.maximum(
#         0.0, cdist(track_features, det_features, metric)
#     )  # Nomalized features

#     return cost_matrix


def embedding_distance_older(tracks: list[STrack],
                             detections: list[STrack]):
    """
    :param tracks: list[STrack]
    :param detections: list[STrack]
    :param metric: str
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [det.emb.get_last() for det in detections], dtype=np.float32
    )
    track_features = np.asarray(
        [track.emb.get() for track in tracks], dtype=np.float32
    )
    cost_matrix = np.maximum(
        0.0, 1 - fastdist.cosine_matrix_to_matrix(track_features, det_features)
    )  # Nomalized features

    return cost_matrix


def embedding_distance(tracks_embs: np.ndarray, dets_embs: np.ndarray):
    cost_matrix = np.zeros(
        (len(tracks_embs), len(dets_embs)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    cost_matrix = np.maximum(
        0.0, 1 - fastdist.cosine_matrix_to_matrix(tracks_embs, dets_embs)
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


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()
    alpha = 0.9

    def __init__(self, det, max_obs, emb=None):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)

        self.conf = det[4]
        self.cls = ClassStorage(det[5], self.conf)
        self.det_ind = det[6]
        self.max_obs = max_obs
        self.emb = EmbeddingHandler(emb) if emb is not None else None
        self.kalman_filter: KalmanFilterXYWH = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # (xc, yc, w, h)

        return xywh2xyxy(ret)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    # @staticmethod
    # def multi_predict(stracks: list[STrack]):
    #     if len(stracks) > 0:
    #         multi_mean = np.asarray([st.mean.copy() for st in stracks])
    #         multi_covariance = np.asarray([st.covariance for st in stracks])
    #         for i, st in enumerate(stracks):
    #             if st.state != TrackState.Tracked:
    #                 multi_mean[i][7] = 0
    #         multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
    #             multi_mean, multi_covariance
    #         )
    #         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
    #             stracks[i].mean = mean
    #             stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self,
                    new_track: STrack,
                    frame_id: int,
                    new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    # def update(self, new_track: STrack, frame_id: int):
    #     """
    #     Update a matched track
    #     :type new_track: STrack
    #     :type frame_id: int
    #     :type update_emb: bool
    #     :return:
    #     """
    #     self.frame_id = frame_id
    #     self.tracklet_len += 1
    #     self.history_observations.append(self.xyxy)

    #     self.mean, self.covariance = self.kalman_filter.update(
    #         self.mean, self.covariance, new_track.xywh
    #     )

    #     if new_track.emb is not None:
    #         self.emb.update(new_track.emb.get_last())

    #     self.state = TrackState.Tracked
    #     self.is_activated = True

    #     self.conf = new_track.conf
    #     self.cls.update(new_track.cls.get(), new_track.conf)
    #     self.det_ind = new_track.det_ind


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
        self.dets_storage = TrackStorage(size=dets_storage_size)
        self.tracks_storage = TrackStorage(size=tracks_storage_size)

        # self.active_tracks: list[STrack] = []
        # self.lost_tracks: list[STrack] = []
        # self.removed_tracks: list[STrack] = []

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
        confs = dets[:, 4]
        inds_first = confs > self.track_high_thresh

        inds_higher_low = confs > self.track_low_thresh
        inds_lower_high = confs < self.track_high_thresh
        inds_second = np.logical_and(inds_higher_low, inds_lower_high)

        dets_high = dets[inds_first]
        dets_low = dets[inds_second]
        embs_high = embs[inds_first] if embs is not None else None
        return dets, dets_high, dets_low, embs_high

    def process_matches(self,
                        tracks_matches_pool: np.ndarray,
                        dets_matches_pool: np.ndarray):

        # Split tracks pool to update and reactivate
        update_tracks_pool = np.intersect1d(
            tracks_matches_pool,
            np.where(self.tracks_storage.states == TrackState.Tracked))

        reactivate_tracks_pool = np.intersect1d(
            tracks_matches_pool,
            np.where(self.tracks_storage.states != TrackState.Tracked))

        # Split dets pool to update and reactivate
        update_dets_pool = dets_matches_pool[np.in1d(tracks_matches_pool,
                                                     update_tracks_pool)]
        reactivate_dets_pool = dets_matches_pool[np.in1d(tracks_matches_pool,
                                                         reactivate_tracks_pool)]

        # Update, reactivate
        self.tracks_storage.update(update_tracks_pool,
                                   self.dets_storage, update_dets_pool,
                                   self.frame_count, self.with_reid)

        self.tracks_storage.reactivate(reactivate_tracks_pool,
                                       self.dets_storage, reactivate_dets_pool,
                                       self.frame_count, self.with_reid)

        return update_tracks_pool, reactivate_tracks_pool

    @staticmethod
    def pool_from_matches(tracks_pool: np.ndarray,
                          dets_pool: np.ndarray,
                          matches: list[tuple[int, int]]):
        tracks_matches, dets_matches = zip(*matches)
        tracks_matches_pool = tracks_pool[tracks_matches]
        dets_matches_pool = dets_pool[dets_matches]
        return tracks_matches_pool, dets_matches_pool

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self,
               dets: np.ndarray,
               img: np.ndarray = None,
               embs: np.ndarray = None) -> np.ndarray:

        self.check_inputs(dets, img)

        self.frame_count += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        activated_pool = np.empty(0, dtype=int)
        reactivated_pool = np.empty(0, dtype=int)
        lost_pool = np.empty(0, dtype=int)
        removed_pool = np.empty(0, dtype=int)

        dets, dets_high, dets_low, embs_high = self._split_detections(dets,
                                                                      embs)

        # Extract appearance embeddings
        if self.with_reid and embs is None:
            embs_high = self.reid_model.get_features(dets_high[:, 0:4], img)
        else:
            embs_high = embs_high if embs_high is not None else None

        """Process detections"""
        dets_high_pool = list(range(len(dets_high)))
        dets_low_pool = list(
            range(len(dets_high), len(dets_high) + len(dets_low)))

        self.dets_storage.set(dets_high_pool, dets_high, embs_high)

        """ Add newly detected tracklets to tracked_stracks"""

        # active tracks, but not activated yet
        unconfirmed_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == False)

        # active and activated
        tracked_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == True)

        """ Step 2: First association, with high conf detection boxes"""
        tracks_pool = np.union1d(tracked_pool, self.lost_pool)

        # Predict the current location with KF
        self.tracks_storage.multi_predict(tracks_pool)

        iou_dists = AssociationFunction.iou_batch(
            xywh2xyxy(self.tracks_storage.dets[tracks_pool]),
            xywh2xyxy(self.dets_storage.dets[dets_high_pool]))

        iou_dists = fuse_score(
            iou_dists, self.dets_storage.confs[dets_high_pool])

        if self.with_reid:
            emb_dists = embedding_distance(
                self.tracks_storage.embs[tracks_pool],
                self.dets_storage.embs[dets_high_pool]) / 2.0

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
        unmatched_dets_high_pool = dets_high_pool[u_dets_high]

        # Update, reactivate
        high_activated_pool, high_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_matches_pool)

        activated_pool = np.union1d(activated_pool, high_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, high_reactivated_pool)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections

        self.dets_storage.set(dets_low_pool, dets_low)

        remain_tracks_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states == TrackState.Tracked)

        iou_dists = AssociationFunction.iou_batch(
            xywh2xyxy(self.tracks_storage.dets[remain_tracks_pool]),
            xywh2xyxy(self.dets_storage.dets[dets_low_pool]))

        matches, u_tracks, u_dets_low = linear_assignment(
            iou_dists, thresh=0.5)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            remain_tracks_pool, dets_low_pool, matches)

        unmatched_tracks_pool = remain_tracks_pool[u_tracks]
        unmatched_dets_low_pool = dets_low_pool[u_dets_low]

        # Update, reactivate
        low_activated_pool, low_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_matches_pool)

        activated_pool = np.union1d(activated_pool, low_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, low_reactivated_pool)

        lost_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states != TrackState.Lost)

        self.tracks_storage.states[lost_pool] = TrackState.Lost

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Unmatched detections from first association (high conf dets)

        iou_dists = AssociationFunction.iou_batch(
            xywh2xyxy(self.tracks_storage.dets[unconfirmed_pool]),
            xywh2xyxy(self.dets_storage.dets[unmatched_dets_high_pool]))

        iou_dists = fuse_score(
            iou_dists, self.dets_storage.confs[unmatched_dets_high_pool])

        matches, u_tracks, u_dets_high = linear_assignment(
            iou_dists, thresh=0.7)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            unconfirmed_pool, unmatched_dets_high_pool, matches)

        unmatched_tracks_pool = unconfirmed_pool[u_tracks]
        unmatched_remain_high_dets_pool = unmatched_dets_high_pool[u_dets_high]

        self.tracks_storage.update(tracks_matches_pool,
                                   self.dets_storage, dets_matches_pool,
                                   self.frame_count, self.with_reid)

        activated_pool = np.union1d(activated_pool, tracks_matches_pool)

        self.tracks_storage.states[unmatched_tracks_pool] = TrackState.Removed

        removed_pool = np.union1d(removed_pool, unmatched_tracks_pool)

        """ Step 4: Init new stracks"""

        new_tracks_pool = np.intersect1d(
            unmatched_remain_high_dets_pool,
            self.dets_storage.confs >= self.track_new_thresh)

        self.tracks_storage.activate(
            self.dets_storage, new_tracks_pool,
            frame_ids=self.frame_count, with_reid=self.with_reid)

        activated_pool = np.union1d(activated_pool,
                                    unmatched_remain_high_dets_pool)

        """ Step 5: Update state"""

        remove_filter = (self.frame_count - self.tracks_storage.start_frames[self.lost_pool] > self.max_time_lost)
        remove_pool = self.lost_pool[remove_filter]

        self.tracks_storage.states[remove_pool] = TrackState.Removed
        removed_pool = np.union1d(removed_pool, remove_pool)

        self.active_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.states == TrackState.Tracked)
        
        self.active_pool = np.union1d(self.active_pool, activated_pool)
        self.active_pool = np.union1d(self.active_pool, reactivated_pool)
        self.active_pool = np.setdiff1d(self.active_pool, removed_pool)

        self.lost_pool = np.setdiff1d(self.lost_pool, self.active_pool)
        self.lost_pool = np.union1d(self.lost_pool, lost_pool)
        self.lost_pool = np.setdiff1d(self.lost_pool, removed_pool)
        
        self.lost_pool.extend(lost_stracks)
        self.lost_pool = sub_stracks(
            self.lost_pool, self.removed_tracks)
        self.removed_tracks.extend(removed_stracks)

        # Clean up
        self.removed_tracks = [
            track for track in self.removed_tracks
            if self.frame_count - track.end_frame < 10 * self.max_time_lost]

        self.active_pool, self.lost_pool = remove_duplicate_stracks(
            self.active_pool, self.lost_pool
        )
        # get confs of lost tracks
        output_stracks = [
            track for track in self.active_pool if track.is_activated]
        outputs = []

        for t in output_stracks:
            output = [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs


# id, class_id, conf


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
