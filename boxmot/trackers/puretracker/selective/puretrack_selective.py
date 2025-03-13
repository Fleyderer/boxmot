from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.puretracker.storage import TrackState, TrackStorage, EmbeddingHandler
from boxmot.trackers.puretracker.utils import (
    xywh2xyxy, xyxy2xywh, linear_assignment,
    embedding_distance, fuse_score, iou_batch_xywh, aiou_batch_xywh)
from boxmot.trackers.basetracker import BaseTracker


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

        iou_dist_risky_thresh: float,
        ars_thresh: float,

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

        self.iou_dist_risky_thresh = iou_dist_risky_thresh
        self.ars_thresh = ars_thresh

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

    def _split_dets(self, dets: np.ndarray, embs: np.ndarray):
        dets = np.hstack((dets, np.arange(len(dets)).reshape(-1, 1)))
        confs = dets[:, 4]

        inds_high = confs > self.track_high_thresh
        inds_low = (confs > self.track_low_thresh) & (confs <= self.track_high_thresh)

        dets_high_xyxy = dets[:, :4][inds_high]
        dets[:, :4] = xyxy2xywh(dets[:, :4])
        dets_high = dets[inds_high]
        dets_low = dets[inds_low]
        embs_high = embs[inds_high] if embs is not None else None

        return dets, dets_high_xyxy, dets_high, dets_low, embs_high

    def process_matches(self,
                        tracks_matches_pool: np.ndarray,
                        dets: np.ndarray, embs: np.ndarray = None,
                        tracks_embs_pool: np.ndarray = None):

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
            
            if tracks_embs_pool is None:
                assert len(embs) == len(dets)
                pass
            
            upd_mask = np.in1d(tracks_embs_pool, update_tracks_pool)
            react_mask = np.in1d(tracks_embs_pool, reactivate_tracks_pool)
            
            update_embs = embs[upd_mask]
            reactivate_embs = embs[react_mask]
            
            tracks_embs_update_pool = tracks_embs_pool[upd_mask]
            tracks_embs_reactivate_pool = tracks_embs_pool[react_mask]

        else:
            update_embs = None
            reactivate_embs = None
            tracks_embs_update_pool = None
            tracks_embs_reactivate_pool = None

        # Update, reactivate
        self.tracks_storage.update(update_tracks_pool, update_dets,
                                   self.frame_count, update_embs,
                                   tracks_embs_update_pool)

        self.tracks_storage.reactivate(reactivate_tracks_pool, reactivate_dets,
                                       self.frame_count, reactivate_embs,
                                       tracks_embs_reactivate_pool)

        return update_tracks_pool, reactivate_tracks_pool

    def iou_distance(self, tracks_pool, dets):

        if len(tracks_pool) == 0 or len(dets) == 0:
            return np.zeros((len(tracks_pool),
                             len(dets)), dtype=np.float32)

        ious = iou_batch_xywh(self.tracks_storage.means[tracks_pool][:, :4],
                              dets)

        return 1 - ious
    
    def aiou_distance(self, tracks_pool, dets):

        if len(tracks_pool) == 0 or len(dets) == 0:
            empty_res = np.zeros((len(tracks_pool), len(dets)), 
                                 dtype=np.float32)
            return empty_res, empty_res

        ious, alphas = aiou_batch_xywh(
            self.tracks_storage.means[tracks_pool][:, :4], dets)

        return 1 - ious, alphas
    
    def split_to_safe_risky(self, iou_dists: np.ndarray, alphas: np.ndarray):

        matches = iou_dists < self.iou_dist_risky_thresh
        match_counts = np.sum(matches, axis=0)

        risky_dets_ids = np.where(match_counts != 1)[0]
        safe_dets_ids = np.where(match_counts == 1)[0]
        candidates = np.argmin(iou_dists[:, safe_dets_ids], axis=0)

        safe_mask = alphas[candidates, safe_dets_ids] > self.ars_thresh

        risky_dets_ids = np.union1d(risky_dets_ids, safe_dets_ids[~safe_mask])
        safe_dets_ids = safe_dets_ids[safe_mask]
        safe_tracks_ids = candidates[safe_mask]

        return safe_tracks_ids, safe_dets_ids, risky_dets_ids

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

        (dets, dets_high_xyxy, dets_high,
         dets_low, embs_high) = self._split_dets(dets, embs)

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

        iou_dists, alphas = self.aiou_distance(tracks_pool, dets_high)


        """ Step 2.1: Filter out detections, where emb extraction is unnecessary"""
        if len(tracks_pool) > 0:
            (safe_track_ids, safe_dets_ids, 
            risky_dets_ids) = self.split_to_safe_risky(iou_dists, alphas)
            
            safe_tracks_pool = tracks_pool[safe_track_ids]
            safe_dets_high_pool = dets_high_pool[safe_dets_ids]
            risky_dets_pool = dets_high_pool[risky_dets_ids]
        else:
            safe_tracks_pool = np.empty(0, dtype=int)
            safe_dets_high_pool = np.empty(0, dtype=int)
            risky_dets_pool = dets_high_pool

        # Extract appearance embeddings only if needed
        if self.with_reid and embs is None:
            embs_risky = self.reid_model.get_features(dets_high_xyxy[risky_dets_pool], img)
        else:
            embs_risky = embs_high[risky_dets_pool] if embs_high is not None else None

        # TODO: maybe move before split_to_safe_risky ^
        iou_dists = fuse_score(
            iou_dists, dets_high[:, 4])

        if self.with_reid:
            emb_risky_dists = embedding_distance(
                self.tracks_storage.embs[tracks_pool],
                embs_risky) / 2.0
            
            emb_dists = np.zeros((len(tracks_pool), 
                                  len(dets_high_pool)), dtype=np.float32)
            emb_dists[:, risky_dets_pool] = emb_risky_dists
            emb_dists[:, safe_dets_high_pool] = 1.0

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

        embs_update_pool = np.intersect1d(dets_matches_pool, risky_dets_pool)
        embs_risky_update = embs_risky[np.in1d(risky_dets_pool, embs_update_pool)]

        # Track ids corresponding to embeddings to be updated
        tracks_embs_update_pool = tracks_matches_pool[
            np.in1d(dets_matches_pool, embs_update_pool)]

        # Update, reactivate
        high_activated_pool, high_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_high[dets_matches_pool], 
            embs_risky_update, tracks_embs_update_pool)

        activated_pool = high_activated_pool
        reactivated_pool = high_reactivated_pool

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
            tracks_matches_pool, dets_low[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, low_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, low_reactivated_pool)

        lost_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states != TrackState.Lost)

        self.tracks_storage.states[lost_pool] = TrackState.Lost

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Unmatched detections from first association (high conf dets)
        dets_high_unmatched = dets_high[unmatched_dets_high_pool]

        iou_dists = self.iou_distance(
            unconfirmed_pool, dets_high_unmatched)

        iou_dists = fuse_score(
            iou_dists, dets_high_unmatched[:, 4])

        # TODO: make another threshold variable if needed
        matches, u_tracks, u_dets_high = linear_assignment(
            iou_dists, thresh=0.7)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            unconfirmed_pool, unmatched_dets_high_pool, matches)

        unmatched_tracks_pool = unconfirmed_pool[u_tracks]
        unmatched_remain_high_dets_pool = unmatched_dets_high_pool[u_dets_high]

        # Comment: using smart update for risky detections we can't provide embeddings here
        # because they were not calculated and can't be, because we get dets_high_unmatched
        # only after few steps, not while extracting features first time.
        self.tracks_storage.update(tracks_matches_pool,
                                   dets_high[dets_matches_pool],
                                   self.frame_count, 
                                   embs_high[dets_matches_pool],
                                   tracks_matches_pool)

        activated_pool = np.union1d(activated_pool, tracks_matches_pool)

        self.tracks_storage.states[unmatched_tracks_pool] = TrackState.Removed

        removed_pool = np.union1d(removed_pool, unmatched_tracks_pool)

        """ Step 4: Init new stracks"""

        dets_to_tracks_pool = np.intersect1d(
            unmatched_remain_high_dets_pool,
            np.where(dets_high[:, 4] >= self.track_new_thresh)[0])
        
        if self.with_reid and embs is None:
            embs_new = self.reid_model.get_features(dets_high_xyxy[dets_to_tracks_pool], img)
        else:
            embs_new = embs_high[dets_to_tracks_pool] if embs_high is not None else None

        new_tracks_pool = self.tracks_storage.activate(
            dets_high[dets_to_tracks_pool], self.frame_count,
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

        self.active_pool, self.lost_pool = self.remove_duplicates(
            self.active_pool, self.lost_pool
        )

        # Clean up removed tracks
        if self.frame_count % self.max_time_lost == 0:
            save_pool = np.union1d(self.active_pool, self.lost_pool)
            self.tracks_storage.cleanup(save_pool)
            self.removed_pool = np.intersect1d(self.removed_pool, save_pool)

        self.removed_pool = np.union1d(self.removed_pool, removed_pool)

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
        """
        Function to remove duplicate tracks, 
        based on their IoU closeness and track duration (longer track wins)

        Args:
            tracksa_pool (np.ndarray): Pool of tracks to be compared
            tracksb_pool (np.ndarray): Pool of tracks to be compared

        Returns:
            tuple: Pools of tracks after duplicates removal
        """
        pdist = 1 - iou_batch_xywh(
            self.tracks_storage.means[tracksa_pool][:, :4],
            self.tracks_storage.means[tracksb_pool][:, :4],
        )

        tracksa_dup, tracksb_dup = np.where(pdist < 0.15)

        # Found duplicates
        tracksa_dup_pool = tracksa_pool[tracksa_dup]
        tracksb_dup_pool = tracksb_pool[tracksb_dup]

        timesa = self.tracks_storage.frame_ids[tracksa_dup_pool] - \
            self.tracks_storage.start_frames[tracksa_dup_pool]
        timesb = self.tracks_storage.frame_ids[tracksb_dup_pool] - \
            self.tracks_storage.start_frames[tracksb_dup_pool]

        # Filter out duplicates, which exists less time
        finala_pool = np.setdiff1d(tracksa_pool,
                                   tracksa_dup_pool[timesa < timesb])
        finalb_pool = np.setdiff1d(tracksb_pool,
                                   tracksb_dup_pool[timesb < timesa])

        return finala_pool, finalb_pool
