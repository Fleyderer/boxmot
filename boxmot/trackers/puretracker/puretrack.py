from __future__ import annotations
from pathlib import Path
import threading

import numpy as np
import cv2
import torch

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.trackers.puretracker.storage import TrackState, TrackStorage, EmbeddingHandler
from boxmot.trackers.puretracker.utils import (
    xywh2xyxy, xyxy2xywh, linear_assignment,
    embedding_distance, fuse_score, iou_batch_xywh, iou_vr_batch_xywh, camera_update)
from boxmot.trackers.puretracker.ecc import ECC
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.puretracker.display import visualize_tracking, show_visualization


class PureTrackNew(BaseTracker):
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

        iou_dist_risky_thresh: float,
        ars_thresh: float,

        tracks_storage_size: int,
        max_time_lost: int,
        max_time_reidable: int = 5.0,
        frame_rate: int = 30,
        per_class: bool = False,

        with_ecc: bool = False,
        with_reid: bool = True,
        with_emb_reactivation: bool = True,
        emb_mode: str = "ema",
        emb_ema_alpha: float = 0.9,
        emb_max_count: int = 10,

        iou_emb_thresh: float = 0.17,
        emb_iou_thresh: float = 0.5,
        emb_thresh: float = 0.5,
        vr_thresh: float = 0.3,
        emb_reid_thresh: float = 0.1,

        cleanup_every: int = 30,

        debug: bool = True,

    ):
        # TODO: Add per class tracking logic for pools
        super().__init__(per_class=per_class)
        self.tracks_storage = TrackStorage(size=tracks_storage_size,
                                           auto_increase=True)

        self.active_pool: np.ndarray = np.empty(0, dtype=int)
        self.lost_pool: np.ndarray = np.empty(0, dtype=int)
        self.reidable_pool: np.ndarray = np.empty(0, dtype=int)
        self.removed_pool: np.ndarray = np.empty(0, dtype=int)

        EmbeddingHandler._mode = emb_mode
        EmbeddingHandler._ema_alpha = emb_ema_alpha
        EmbeddingHandler._max_len = emb_max_count

        self.frame_count = 0
        self.per_class = per_class

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_new_thresh = track_new_thresh
        self.match_thresh = match_thresh

        self.max_time_lost = max_time_lost
        self.frame_rate = frame_rate
        self._max_frames_lost = int(max_time_lost * frame_rate)

        self.max_time_reidable = max_time_reidable
        self._max_frames_reidable = int(max_time_reidable * frame_rate)

        self.with_ecc = with_ecc
        self.with_reid = with_reid
        self.with_emb_reactivation = with_emb_reactivation
        self.debug = debug

        if self.with_reid:
            self.reid_model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half).model
        else:
            self.reid_model = None

        if self.with_ecc:
            self.ecc = ECC(scale=350, video_name=None, use_cache=True)
        else:
            self.ecc = None

        self.emb_iou_thresh = emb_iou_thresh
        self.iou_emb_thresh = iou_emb_thresh
        self.emb_thresh = emb_thresh
        self.vr_thresh = vr_thresh
        self.emb_reid_thresh = emb_reid_thresh

        self.cleanup_every = cleanup_every

        self._debug = {}

    def _split_dets(self, dets: np.ndarray, embs: np.ndarray):
        dets = np.hstack((dets, np.arange(len(dets)).reshape(-1, 1)))

        confs = dets[:, 4]

        inds_high = confs > self.track_high_thresh
        inds_low = (confs > self.track_low_thresh) & (
            confs <= self.track_high_thresh)

        dets_high_xyxy = dets[:, :4][inds_high]
        dets[:, :4] = xyxy2xywh(dets[:, :4])
        dets_high = dets[inds_high]
        dets_low = dets[inds_low]
        embs_high = embs[inds_high] if embs is not None else None

        # TODO remove
        confs_high, confs_low = confs[inds_high], confs[inds_low]

        return dets, dets_high_xyxy, dets_high, dets_low, embs_high, confs_high, confs_low

    def apply_ecc(self, img: np.ndarray, tracks_pool: np.ndarray):
        transform = self.ecc(img, self.frame_count)

        if tracks_pool.size > 0:
            # Get the current means
            means = self.tracks_storage.means[tracks_pool]
            # Update the position part
            means[:, :4] = camera_update(means[:, :4], transform)
            # Write back the entire array, not just a slice
            self.tracks_storage.means[tracks_pool] = means

    def process_matches(self,
                        tracks_matches_pool: np.ndarray,
                        dets: np.ndarray, embs: np.ndarray = None,
                        pure_tracks_pool: np.ndarray = None):

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

        if update_embs is not None and pure_tracks_pool is not None:
            update_pure_tracks_filter = np.isin(update_tracks_pool, 
                                                pure_tracks_pool)
            update_pure_tracks_pool = update_tracks_pool[update_pure_tracks_filter]
            update_pure_embs = update_embs[update_pure_tracks_filter]
        else:
            update_pure_tracks_pool = None
            update_pure_embs = None

        # Update, reactivate
        self.tracks_storage.update(update_tracks_pool, update_dets,
                                   self.frame_count, update_embs,
                                   update_pure_tracks_pool, update_pure_embs)

        # TODO: Check is it better to update or replace or do not change embs in reactivation here.
        self.tracks_storage.reactivate(reactivate_tracks_pool, reactivate_dets,
                                       self.frame_count)

        return update_tracks_pool, reactivate_tracks_pool

    def iou_distance(self, tracks_pool, dets, calc_vr: bool = False):

        if len(tracks_pool) == 0 or len(dets) == 0:
            iou_dists = np.zeros((len(tracks_pool),
                                  len(dets)), dtype=np.float32)
            vrs = None
        else:
            tracks = self.tracks_storage.means[tracks_pool][:, :4]

            if calc_vr:
                ious, vrs = iou_vr_batch_xywh(tracks, dets)
            else:
                ious = iou_batch_xywh(tracks, dets)
                vrs = None
                
            iou_dists = 1 - ious
        
        return (iou_dists, vrs) if calc_vr else iou_dists
                
        
    @staticmethod
    def pool_from_matches(tracks_pool: np.ndarray, dets_pool: np.ndarray,
                          matches: list[tuple[int, int]]):
        if len(matches) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        tracks_matches, dets_matches = zip(*matches)
        tracks_matches_pool = tracks_pool[list(tracks_matches)]
        dets_matches_pool = dets_pool[list(dets_matches)]
        return tracks_matches_pool, dets_matches_pool
    

    def update_state(self, 
                     removed_pool: np.ndarray,
                     activated_pool: np.ndarray, 
                     reactivated_pool: np.ndarray,
                     lost_pool: np.ndarray) -> np.ndarray:
        # --- Step 1: Determine removal candidates based on max frame threshold ---
        max_frames_filter = (
                self.frame_count - self.tracks_storage.frame_ids[self.lost_pool] > self._max_frames_lost)
            
        
        if self.with_emb_reactivation:
            # Identify lost tracks eligible for reactivation.
            reidable_pool = self.lost_pool[max_frames_filter]
            self.tracks_storage.states[reidable_pool] = TrackState.Reidable
            self.tracks_storage.frame_ids[reidable_pool] = self.frame_count

            # Determine which tracks in the reidable pool have exceeded reactivation threshold.
            remove_filter = (
                self.frame_count - self.tracks_storage.frame_ids[self.reidable_pool] > self._max_frames_reidable)
            remove_pool = self.reidable_pool[remove_filter]
        
        else:
            reidable_pool = np.empty(0, dtype=int)
            remove_pool = self.lost_pool[max_frames_filter]

        
        if self.debug and 7 in lost_pool:
            print('HERE!')

        # Remove these tracks
        self.tracks_storage.states[remove_pool] = TrackState.Removed
        removed_pool = np.union1d(removed_pool, remove_pool)
        self.removed_pool = np.union1d(self.removed_pool, removed_pool)

        # --- Step 2: Update Active Pool ---
        self.active_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.states == TrackState.Tracked)
        self.active_pool = np.union1d(self.active_pool, activated_pool)
        self.active_pool = np.union1d(self.active_pool, reactivated_pool)

        # --- Step 3: Update Lost Pool ---
        self.lost_pool = np.setdiff1d(self.lost_pool, self.active_pool)
        self.lost_pool = np.union1d(self.lost_pool, lost_pool)
        self.lost_pool = np.setdiff1d(self.lost_pool, reidable_pool)
        self.lost_pool = np.setdiff1d(self.lost_pool, self.removed_pool)
        
        # --- Step 4: Update Reidable Pool (optional) ---
        if self.with_emb_reactivation:
            self.reidable_pool = np.setdiff1d(self.reidable_pool, self.active_pool)
            self.reidable_pool = np.union1d(self.reidable_pool, reidable_pool)
            self.reidable_pool = np.setdiff1d(self.reidable_pool, self.removed_pool)

        # --- Step 5: Remove Duplicates ---
        self.active_pool, self.lost_pool = self.remove_duplicates(
            self.active_pool, self.lost_pool
        )

        # --- Step 6: Periodic Cleanup ---
        if self.frame_count % self.cleanup_every == 0:
            save_pool = np.union1d(self.active_pool, self.lost_pool)
            save_pool = np.union1d(save_pool, self.reidable_pool)

            self.tracks_storage.cleanup(save_pool)
            self.removed_pool = np.intersect1d(self.removed_pool, save_pool)

        # --- Step 7: Compute Output ---
        output_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == True)
        
        return output_pool

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
         dets_low, embs_high, confs_high, confs_low) = self._split_dets(dets, embs)

        # Extract appearance embeddings
        if self.with_reid and embs is None:
            embs_high = self.reid_model.get_features(dets_high_xyxy, img)
        else:
            embs_high = embs_high if embs_high is not None else None

        """ Add newly detected tracklets to tracked_stracks"""

        # active tracks, but not activated yet
        unconfirmed_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == False)

        # active and activated
        tracked_pool = np.intersect1d(
            self.active_pool, self.tracks_storage.is_activated_tracks == True)
        
        """ Step 1: Reidentification of tracks (optional) """

        if self.with_reid and self.with_emb_reactivation and self.reidable_pool.size > 0:

            reid_tracks_pool = self.reidable_pool
            dets_high_pool = np.arange(len(dets_high))

            # TODO: Needed if we will use IOU, else remove
            # self.tracks_storage.multi_predict(reid_tracks_pool)

            if self.ecc is not None:
                self.apply_ecc(img, reid_tracks_pool)

            emb_dists = embedding_distance(
                self.tracks_storage.pure_embs[reid_tracks_pool],
                embs_high) / 2.0
            
            # TODO: think of another way to match, because if there are
            # more than one candidate, we can decide not to choosing any of them
            matches, u_tracks, u_dets_high = linear_assignment(
            emb_dists, thresh=self.emb_reid_thresh)

            tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
                reid_tracks_pool, dets_high_pool, matches)

            if len(matches) > 0:
                print('HERE match!')

            self.tracks_storage.reactivate(
                tracks_matches_pool, dets_high[dets_matches_pool],
                self.frame_count, embs_high[dets_matches_pool])
            
            reactivated_pool = np.union1d(reactivated_pool, tracks_matches_pool)

            dets_high = dets_high[u_dets_high]
            embs_high = embs_high[u_dets_high]

        """ Step 2: First association, with high conf detection boxes"""
        tracks_pool = np.union1d(tracked_pool, self.lost_pool)
        dets_high_pool = np.arange(len(dets_high))

        # Predict the current location with KF
        self.tracks_storage.multi_predict(tracks_pool)

        # Update ECC
        if self.ecc is not None:
            self.apply_ecc(img, tracks_pool)

        iou_dists, vrs = self.iou_distance(tracks_pool, dets_high,
                                                calc_vr=True)

        iou_dists = fuse_score(
            iou_dists, dets_high[:, 4])

        if self.with_reid:
            emb_dists = embedding_distance(
                self.tracks_storage.embs[tracks_pool],
                embs_high) / 2.0

            if self.debug and self.frame_count == 207:
                print('HERE!!!')

            iou_dists[emb_dists > self.iou_emb_thresh] = 1.0
            emb_dists[emb_dists > self.emb_thresh] = 1.0
            emb_dists[iou_dists > self.emb_iou_thresh] = 1.0
            dists = np.minimum(iou_dists, emb_dists)
        else:
            dists = iou_dists

        # Matches, unmatched tracks ids, unmatched detections ids
        matches, u_tracks, u_dets_high = linear_assignment(
            dists, thresh=self.match_thresh)

        if self.debug:
            self._debug["img"] = img.copy()

            if len(tracks_pool) > 0:
                self._debug["track_ids_high"] = tracks_pool.astype(int)
                self._debug["tracks_high"] = self.tracks_storage.means[tracks_pool][:, :4]
                self._debug["dets_ids_high"] = dets_high_pool
                self._debug["dets_high"] = dets_high[:, :4]
                self._debug["dets_high_confs"] = confs_high

                self._debug["iou_dists_high"] = iou_dists
                self._debug["matches_high"] = matches

                if self.with_reid:
                    self._debug["emb_dists_high"] = emb_dists

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            tracks_pool, dets_high_pool, matches)

        if self.debug and 2 in tracks_matches_pool:
            det_id = dets_matches_pool[np.argwhere(
                tracks_matches_pool == 2)[0][0]]
            closest_iou = np.partition(dists[:, det_id], 2)[1]
            closest_vr = np.partition(vrs[:, det_id], 2)[1] if vrs is not None else -1.0
            cur_emb = np.expand_dims(embs_high[det_id], 0)

            print(f"{closest_iou:.2f}, {closest_vr:.2f}")

            if closest_vr > 0.3:
                if not "clean_emb" in self._debug:
                    self._debug["clean_emb"] = cur_emb
                else:
                    self._debug["clean_emb"] = EmbeddingHandler.update(
                        self._debug["clean_emb"], cur_emb)
                self._debug["last_clean_emb"] = embs_high[det_id]
            else:
                pass

            self._debug["last_emb"] = embs_high[det_id]
        elif self.debug and 2 in self.lost_pool:
            pass


        if self.with_emb_reactivation and vrs is not None:
            # Find second closest Visual Ratios to determine 
            # the visibility of an object
            closest_vrs = np.partition(vrs, 1, axis=0)[1, :]
            pure_dets_pool = dets_high_pool[closest_vrs > self.vr_thresh]
            pure_tracks_pool = tracks_matches_pool[np.isin(dets_matches_pool, 
                                                           pure_dets_pool)]
        else:
            pure_tracks_pool = None

            
        unmatched_tracks_pool = tracks_pool[u_tracks]
        unmatched_dets_high_pool = dets_high_pool[list(u_dets_high)]

        # Update, reactivate
        high_activated_pool, high_reactivated_pool = self.process_matches(
            tracks_matches_pool, 
            dets_high[dets_matches_pool], embs_high[dets_matches_pool],
            pure_tracks_pool)

        activated_pool = np.union1d(activated_pool, high_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, high_reactivated_pool)

        if self.debug:

            if self._debug.get("tracks_high") is not None:

                d_img, cell_positions, track_crops, detection_crops = visualize_tracking(
                    img=self._debug["img"], t_boxes=self._debug["tracks_high"],
                    d_boxes=self._debug["dets_high"], dets_confs=self._debug["dets_high_confs"],
                    iou_dists=self._debug["iou_dists_high"], emb_dists=self._debug["emb_dists_high"],
                    title="High matching step", t_ids=self._debug["track_ids_high"])

                show_visualization(d_img, cell_positions, track_crops, detection_crops,
                                   frame_num=self.frame_count, window_name="High matching step")

        """ Step 3: Second association, with low conf detection boxes"""
        # association of the unmatched but tracked to the low conf detections

        remain_tracks_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states == TrackState.Tracked)

        if self.debug and self.frame_count >= 172:
            print('HERE!')

        if self.debug and 7 in self.tracks_storage._manager._track_to_storage:
            print("HERE!!!")

        dets_low_pool = np.arange(len(dets_low))

        iou_dists = self.iou_distance(remain_tracks_pool, dets_low)

        matches, u_tracks, u_dets_low = linear_assignment(
            iou_dists, thresh=0.5)

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            remain_tracks_pool, dets_low_pool, matches)

        if self.debug and len(remain_tracks_pool) > 0:
            self._debug["track_ids_low"] = remain_tracks_pool.astype(int)
            self._debug["tracks_low"] = self.tracks_storage.means[remain_tracks_pool][:, :4]
            self._debug["dets_low"] = dets_low[:, :4]
            self._debug["dets_ids_low"] = dets_low_pool
            self._debug["dets_low_confs"] = confs_low
            self._debug["iou_dists_low"] = iou_dists
            self._debug["matches_low"] = list(
                zip(tracks_matches_pool, dets_matches_pool))

        unmatched_tracks_pool = remain_tracks_pool[u_tracks]
        unmatched_dets_low_pool = u_dets_low

        # Update, reactivate
        low_activated_pool, low_reactivated_pool = self.process_matches(
            tracks_matches_pool, dets_low[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, low_activated_pool)
        reactivated_pool = np.union1d(reactivated_pool, low_reactivated_pool)

        # TODO remove unnecessary logic, because we can just:
        # self.tracks_storage.states[unmatched_tracks_pool] = TrackState.Lost
        lost_pool = np.intersect1d(
            unmatched_tracks_pool,
            self.tracks_storage.states != TrackState.Lost)

        self.tracks_storage.states[lost_pool] = TrackState.Lost

        if self.debug:

            if self._debug.get("tracks_low") is not None:
                dl_img, cell_positions, track_crops, detection_crops = visualize_tracking(
                    img=self._debug["img"], t_boxes=self._debug["tracks_low"],
                    d_boxes=self._debug["dets_low"], dets_confs=self._debug["dets_low_confs"],
                    iou_dists=self._debug["iou_dists_low"], emb_dists=None,
                    title="Low matching step", t_ids=self._debug["track_ids_low"])

                show_visualization(dl_img, cell_positions, track_crops, detection_crops,
                                   frame_num=self.frame_count, window_name="Low matching step")

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        # Unmatched detections from first association (high conf dets)
        dets_high_unmatched = dets_high[unmatched_dets_high_pool]

        iou_dists = self.iou_distance(
            unconfirmed_pool, dets_high_unmatched)

        iou_dists = fuse_score(
            iou_dists, dets_high_unmatched[:, 4])

        # TODO: Add this threshold to hyperparams
        matches, u_tracks, u_dets_high = linear_assignment(
            iou_dists, thresh=0.7)

        if self.debug:
            self._debug["dets_high_unmatched"] = dets_high_unmatched
            self._debug["iou_dists_high_unmatched"] = iou_dists
            self._debug["matches_high_unmatched"] = matches

        tracks_matches_pool, dets_matches_pool = self.pool_from_matches(
            unconfirmed_pool, unmatched_dets_high_pool, matches)

        unmatched_tracks_pool = unconfirmed_pool[u_tracks]
        unmatched_remain_high_dets_pool = unmatched_dets_high_pool[u_dets_high]

        self.tracks_storage.update(tracks_matches_pool,
                                   dets_high[dets_matches_pool],
                                   self.frame_count, embs_high[dets_matches_pool])

        activated_pool = np.union1d(activated_pool, tracks_matches_pool)

        # TODO: Check if Removed better than Reidable in case of unconfirmed tracks
        # 80% that Removed is better
        self.tracks_storage.states[unmatched_tracks_pool] = TrackState.Removed

        removed_pool = np.union1d(removed_pool, unmatched_tracks_pool)

        """ Step 4: Init new stracks"""

        dets_to_tracks_pool = np.intersect1d(
            unmatched_remain_high_dets_pool,
            np.where(dets_high[:, 4] >= self.track_new_thresh)[0])

        new_tracks_pool = self.tracks_storage.activate(
            dets_high[dets_to_tracks_pool], self.frame_count,
            embs_high[dets_to_tracks_pool])

        activated_pool = np.union1d(activated_pool, new_tracks_pool)

        """ Step 5: Update state"""

        output_pool = self.update_state(removed_pool=removed_pool,
                                        activated_pool=activated_pool, 
                                        reactivated_pool=reactivated_pool, 
                                        lost_pool=lost_pool)

        if len(output_pool) > 0:

            xyxys = xywh2xyxy(self.tracks_storage.means[output_pool][:, :4])
            ids = output_pool.reshape(-1, 1)
            confs = self.tracks_storage.confs[output_pool].reshape(-1, 1)
            classes = self.tracks_storage.classes[output_pool].reshape(-1, 1)
            det_ids = self.tracks_storage.det_ids[output_pool].reshape(-1, 1)

            outputs = np.hstack((xyxys, ids, confs, classes, det_ids))
        else:
            return np.empty((0, 6), dtype=np.float32)

        if self.debug and self.frame_count >= 200:
            print("MEANS:", self.tracks_storage.means[2])
            print("LOST:", self.lost_pool)
            print("REMOVED:", self.removed_pool)
            print("ACTIVE:", self.active_pool)
            print("TRACK:", self.tracks_storage.states[2])

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
        if len(tracksa_pool) == 0 or len(tracksb_pool) == 0:
            return tracksa_pool, tracksb_pool

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

    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # TODO: Add plotting for per class
        if self.per_class_active_tracks is not None:
            raise NotImplementedError("Not implemented yet.")

        else:
            ids = self.active_pool
            confs = self.tracks_storage.confs[ids]
            classes = self.tracks_storage.classes[ids]

            for idx, conf, cls in zip(ids, confs, classes):
                history = self.tracks_storage.history.get(idx, [])
                if len(history) > 2:
                    history = xywh2xyxy(np.array(history))
                    box = history[-1]

                    img = self.plot_box_on_img(img, box, conf, cls,
                                               idx, thickness, fontscale)

                    if show_trajectories:
                        img = self.plot_trackers_trajectories(
                            img, history, idx)

        return img
