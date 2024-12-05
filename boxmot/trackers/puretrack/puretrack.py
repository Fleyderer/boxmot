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
from boxmot.trackers.puretrack.basetrack import (BaseTrack, TrackState,
                                                 EmbeddingStorage,
                                                 ClassStorage)
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import xyxy2xywh, xywh2xyxy
from boxmot.trackers.basetracker import BaseTracker


def embedding_distance(tracks: list[STrack],
                       detections: list[STrack],
                       metric='cosine'):
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
        0.0, cdist(track_features, det_features, metric)
    )  # Nomalized features

    return cost_matrix


def embedding_distance_fast(tracks: list[STrack],
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
        self.emb = EmbeddingStorage(emb) if emb is not None else None
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

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = self.shared_kalman
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

    def update(self, new_track: STrack, frame_id: int):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_emb: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )

        if new_track.emb is not None:
            self.emb.update(new_track.emb.get_last())

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls.update(new_track.cls.get(), new_track.conf)
        self.det_ind = new_track.det_ind


class PureTrack(BaseTracker):
    """
    PureTracker: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

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
        self.active_tracks: list[STrack] = []  # type: list[STrack]
        self.lost_stracks: list[STrack] = []  # type: list[STrack]
        self.removed_stracks: list[STrack] = []  # type: list[STrack]

        EmbeddingStorage._mode = emb_mode
        EmbeddingStorage._ema_alpha = emb_ema_alpha
        EmbeddingStorage._max_len = emb_max_count

        self.frame_count = 0
        self.track_buffer = track_buffer
        self.per_class = per_class

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_new_thresh = track_new_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = STrack.shared_kalman

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

        dets, dets_high, dets_low, embs_high = self._split_detections(dets,
                                                                      embs)

        # Extract appearance embeddings
        if self.with_reid and embs is None:
            embs_high = self.reid_model.get_features(dets_high[:, 0:4], img)
        else:
            embs_high = embs_high if embs_high is not None else []

        if len(dets_high) > 0:
            """Detections"""
            if self.with_reid:
                detections = [
                    STrack(dets_high[i], max_obs=self.max_obs,
                           emb=embs_high[i])
                    for i in range(len(dets_high))
                ]
            else:
                detections = [
                    STrack(dets_high[i], max_obs=self.max_obs)
                    for i in range(len(dets_high))
                ]
        else:
            detections = []

        # DEBUG
        dets_check = get_tracks(detections)

        """ Add newly detected tracklets to tracked_stracks"""
        # active tracks, but not activated yet
        unconfirmed: list[STrack] = []
        # active and activated
        tracked_stracks: list[STrack] = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # DEBUG
        unconf_check = get_tracks(unconfirmed)
        tracked_check = get_tracks(tracked_stracks)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        iou_dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        iou_dists = fuse_score(iou_dists, detections)

        # DEBUG
        strack_pool_check = get_tracks(strack_pool)

        if self.with_reid:
            emb_dists = embedding_distance_fast(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.emb_thresh] = 1.0
            emb_dists[iou_dists > self.iou_thresh] = 1.0
            dists = np.minimum(iou_dists, emb_dists)
        else:
            dists = iou_dists


        # Matches, unmatched tracks, unmatched detections
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track: STrack = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # DEBUG
        activated_check = get_tracks(activated_stracks)
        refind_check = get_tracks(refind_stracks)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_low) > 0:
            """Detections"""
            detections_second = [
                STrack(det_second, max_obs=self.max_obs) for det_second in dets_low]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        iou_dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(
            iou_dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Unmatched detections from first association (high conf dets)
        detections = [detections[i] for i in u_detection]
        iou_dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        iou_dists = fuse_score(iou_dists, detections)
        matches, u_track, u_detection = linear_assignment(
            iou_dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_track:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.track_new_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]

        self.active_tracks = joint_stracks(
            self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Clean up
        self.removed_stracks = [
            track for track in self.removed_stracks
            if self.frame_count - track.end_frame < 10 * self.max_time_lost]
        

        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        # get confs of lost tracks
        output_stracks = [
            track for track in self.active_tracks if track.is_activated]
        outputs = []

        for t in output_stracks:
            output = [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            outputs.append(output)

        outputs = np.asarray(outputs)
        return outputs


# id, class_id, conf


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


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

def get_tracks(tracks: list[STrack]):
    res_tracks = []
    for track in tracks:
        if hasattr(track, "id"):
            res_tracks.append([*track.xyxy, track.id, track.conf, track.cls])
        else:
            res_tracks.append([*track.xyxy, -1, track.conf, track.cls])
    return res_tracks

def get_ids(tracks: list[STrack]):
    return [track.id for track in tracks]