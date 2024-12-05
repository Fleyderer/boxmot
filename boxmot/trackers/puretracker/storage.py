from __future__ import annotations
from typing import List, Any, Union
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.utils.ops import xyxy2xywh

import numpy as np


class EmbeddingHandler:
    _mode = None
    _ema_alpha = None
    _max_len = None

    @classmethod
    def update(cls, prev_embs: np.ndarray, new_embs: np.ndarray):
        """
        Update embeddings

        Parameters:
        ---------------
        prev_embs: np.ndarray
            Previous embeddings of shape [N, D], where
            N - number of embeddings
            D - embedding shape
        new_embs: np.ndarray
            New embeddings of shape [N, D]
        """

        new_embs /= np.linalg.norm(new_embs, axis=1)[:, np.newaxis]

        if cls._mode == "ema":
            embs = (prev_embs * cls._ema_alpha +
                    new_embs * (1 - cls._ema_alpha))
        elif cls._mode == "last":
            embs = new_embs

        return embs

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackStorageManager:
    """
    Class to manage storage for tracks
    """

    def __init__(self, size: int):
        # Dictionary to map track IDs to storage positions
        self._track_to_storage = {}
        # And vice versa
        self._storage_to_track = {}
        # Set to keep track of free storage positions
        self._free_indices = set(range(size))
        self._max_id = 0
        self._size = size

    def is_full(self):
        return not self._free_indices

    def space_left(self):
        return len(self._free_indices)

    def increase_size(self, size: int):
        self._free_indices = self._free_indices | set(range(self._size, size))
        self._size = size

    def get_max_id(self):
        return self._max_id

    def add(self, track_id: int):
        if track_id in self._track_to_storage:
            raise ValueError(f"Track ID {track_id} already exists.")

        if track_id > self._max_id:
            self._max_id = track_id

        if self.is_full():
            raise ValueError("No free storage positions available.")

        # Find a free index
        index = self._free_indices.pop()
        self._track_to_storage[track_id] = index
        self._storage_to_track[index] = track_id

    def cleanup(self, save_ids: list[int]):
        for track_id in list(self._track_to_storage.keys()):
            if track_id not in save_ids:
                self.remove(track_id)

    def remove(self, track_id: int):
        if track_id not in self._track_to_storage:
            raise ValueError(f"Track ID {track_id} does not exist.")

        # Get the index of the track to be removed
        index = self._track_to_storage.pop(track_id)
        self._storage_to_track.pop(index)
        self._free_indices.add(index)

    def get_storage_ids(self,
                        track_ids: list[int],
                        create_new: bool) -> list[int]:
        ids = []

        for track_id in track_ids:
            if track_id not in self._track_to_storage:
                if create_new:
                    self.add(track_id)
                else:
                    raise ValueError(f"Track ID {track_id} does not exist.")

            ids.append(self._track_to_storage[track_id])

        return ids

    def get_track_ids(self, storage_ids: list[int] = None) -> list[int]:
        if storage_ids is None:
            storage_ids = self._track_to_storage.keys()

        return [self._storage_to_track[storage_id]
                for storage_id in storage_ids]


class TrackStorageProperty:
    def __init__(self, name: str):
        self.name = name

    def __get__(self,
                instance: 'TrackStorage',
                owner: type) -> 'TrackStoragePropertyAccessor':
        if instance is None:
            return self
        return TrackStoragePropertyAccessor(instance, self.name)

    def __set__(self, instance: 'TrackStorage', value: Any):
        raise AttributeError(f"Cannot set attribute '{self.name}' directly")


class TrackStoragePropertyAccessor:
    def __init__(self, storage: 'TrackStorage', property_name: str):
        self.storage = storage
        self.property_name = property_name

    def __getattr__(self, name):
        return self.storage._data_dict[self.property_name].__getattribute__(name)

    def __getitem__(self, track_ids: Union[int, List[int]]
                    ) -> Union[Any, np.ndarray]:
        st_indices = self.storage.translate_track_ids(track_ids)

        if self.property_name not in self.storage._data_dict:
            return np.array([])

        return self.storage._data_dict[self.property_name][st_indices]

    def __setitem__(self,
                    track_ids: Union[int, List[int]],
                    values: Union[Any, np.ndarray]):
        st_indices = self.storage.translate_track_ids(track_ids,
                                                      create_new=True)

        if self.property_name not in self.storage._data_dict:

            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=type(values))

            self.storage._data_dict[self.property_name] = np.zeros(
                (self.storage._size, *values.shape[1:]), dtype=values.dtype)

        self.storage._data_dict[self.property_name][st_indices] = values

    def _compare(self, value, comp_method: str):
        items = list(self.storage._manager._track_to_storage.items())

        if len(items) == 0:
            return np.array([])

        track_ids, storage_ids = list(zip(*items))

        data = self.storage._data_dict[self.property_name][list(storage_ids)]

        return np.array(track_ids)[getattr(data, comp_method)(value)]

    def __eq__(self, value):
        return self._compare(value, "__eq__")

    def __ne__(self, value):
        return self._compare(value, "__ne__")

    def __lt__(self, value):
        return self._compare(value, "__lt__")

    def __le__(self, value):
        return self._compare(value, "__le__")

    def __gt__(self, value):
        return self._compare(value, "__gt__")

    def __ge__(self, value):
        return self._compare(value, "__ge__")

    def __add__(self, value):
        return self.storage._data_dict[self.property_name] + value

    def __sub__(self, value):
        return self.storage._data_dict[self.property_name] - value

    def __invert__(self):
        return self.storage._data_dict[self.property_name].__invert__()

    def __len__(self):
        return self.storage._data_dict[self.property_name].__len__()

    def __str__(self):
        return self.storage._data_dict[self.property_name].__str__()

    def __repr__(self):
        return self.storage._data_dict[self.property_name].__repr__()


class TrackStorage:
    dets: np.ndarray[float] = TrackStorageProperty('dets')  # xcycwh
    confs: np.ndarray[float] = TrackStorageProperty('confs')
    classes: np.ndarray[int] = TrackStorageProperty('classes')
    det_ids: np.ndarray[int] = TrackStorageProperty('det_ids')
    means: np.ndarray[float] = TrackStorageProperty('means')
    covs: np.ndarray[float] = TrackStorageProperty('covs')
    embs: np.ndarray[float] = TrackStorageProperty('embs')
    emb_handler: EmbeddingHandler = None
    kalman_filter = KalmanFilterXYWH()
    
    # TODO: Find where it is used and change values
    max_obs: int = None
    states: np.ndarray[int] = TrackStorageProperty('states')
    is_activated_tracks: np.ndarray[bool] = TrackStorageProperty(
        'is_activated_tracks')
    tracklet_lens: np.ndarray[int] = TrackStorageProperty('tracklet_lens')
    frame_ids = TrackStorageProperty('frame_ids')
    start_frames = TrackStorageProperty('start_frames')

    # TODO: Check if this is the correct type or maybe remove prop completely
    # history_observations: np.ndarray[float] = TrackStorageProperty(
    #     'history_observations')

    def __init__(self, size: int, auto_increase: bool = False):
        self._data_dict = {}
        self._size = size
        self._auto_increase = auto_increase
        self._manager = TrackStorageManager(size=self._size)

    def _increase_size(self):
        self._size *= 2
        self._manager.increase_size(size=self._size)
        for key in self._data_dict:
            self._data_dict[key] = np.concatenate(
                (self._data_dict[key],
                 np.zeros(self._data_dict[key].shape,
                          dtype=self._data_dict[key].dtype)), axis=0)

    def translate_track_ids(self,
                            track_ids: Union[int, List[int], np.ndarray],
                            create_new: bool = False
                            ) -> Union[int, List[int]]:

        if isinstance(track_ids, int):
            track_ids = [track_ids]

        if self._auto_increase and len(track_ids) > self._manager.space_left():
            # print(len(track_ids), self._manager.space_left(), self._size)
            self._increase_size()

        return self._manager.get_storage_ids(track_ids, create_new=create_new)

    def remove(self, track_ids: Union[int, List[int]]):
        if isinstance(track_ids, int):
            track_ids = [track_ids]

        for track_id in track_ids:
            self._manager.remove(track_id)

    def multi_predict(self, pool: np.ndarray):
        multi_mean = self.means[pool].copy()
        multi_covariance = self.covs[pool].copy()

        if multi_mean.shape[0] == 0:
            return

        multi_mean[self.states[pool] != TrackState.Tracked, 7] = 0
        multi_mean, multi_covariance = self.kalman_filter.multi_predict(
            multi_mean, multi_covariance)

        self.means[pool] = multi_mean
        self.covs[pool] = multi_covariance

    def set(self, tracks_pool: np.ndarray,
            dets: np.ndarray, embs: np.ndarray = None):
        self.dets[tracks_pool] = xyxy2xywh(dets[:, 0:4])
        self.confs[tracks_pool] = dets[:, 4]
        self.classes[tracks_pool] = dets[:, 5]
        self.det_ids[tracks_pool] = dets[:, 6]
        if embs is not None:
            self.embs[tracks_pool] = embs
        self.means[tracks_pool] = np.nan
        self.covs[tracks_pool] = np.nan
        self.states[tracks_pool] = TrackState.New
        self.is_activated_tracks[tracks_pool] = False
        self.tracklet_lens[tracks_pool] = 0
        self.frame_ids[tracks_pool] = 0

    def from_dets(self, tracks_pool: np.ndarray,
                  dets_storage: TrackStorage, dets_pool: np.ndarray,
                  with_reid: bool):
        self.dets[tracks_pool] = dets_storage.dets[dets_pool]
        self.confs[tracks_pool] = dets_storage.confs[dets_pool]
        self.classes[tracks_pool] = dets_storage.classes[dets_pool]
        self.det_ids[tracks_pool] = dets_storage.det_ids[dets_pool]

        if with_reid:
            self.embs[tracks_pool] = dets_storage.embs[dets_pool]

    def update(self, track_pool: np.ndarray,
               dets_storage: TrackStorage, dets_pool: np.ndarray,
               frame_ids: int | np.ndarray, with_reid: bool):
        if track_pool.shape[0] == 0 or dets_pool.shape[0] == 0:
            return

        self.frame_ids[track_pool] = frame_ids
        self.tracklet_lens[track_pool] += 1
        means, covs = list(zip(*[
            self.kalman_filter.update(
                self.means[track_pool][i],
                self.covs[track_pool][i],
                dets_storage.dets[dets_pool][i])
            for i in range(len(track_pool))]))

        self.means[track_pool] = np.array(means)
        self.covs[track_pool] = np.array(covs)

        if with_reid:
            self.embs[track_pool] = EmbeddingHandler.update(
                self.embs[track_pool], dets_storage.embs[dets_pool])

        self.confs[track_pool] = dets_storage.confs[dets_pool]
        self.classes[track_pool] = dets_storage.classes[dets_pool]
        self.det_ids[track_pool] = dets_storage.det_ids[dets_pool]
        self.is_activated_tracks[track_pool] = True

    def activate(self, dets_storage: TrackStorage, dets_pool: np.ndarray,
                 frame_ids: int | np.ndarray, with_reid: bool):
        """Start a new tracklet"""
        if dets_pool.shape[0] == 0:
            return dets_pool

        start_id = self._manager.get_max_id() + 1
        tracks_pool = range(start_id, start_id + dets_pool.shape[0])

        self.from_dets(tracks_pool, dets_storage, dets_pool, with_reid)

        means, covs = list(zip(*[
            self.kalman_filter.initiate(dets_storage.dets[dets_pool][i])
            for i in range(len(dets_pool))]))

        self.means[tracks_pool] = np.array(means)
        self.covs[tracks_pool] = np.array(covs)

        self.frame_ids[tracks_pool] = frame_ids
        self.tracklet_lens[tracks_pool] = 0
        self.states[tracks_pool] = TrackState.Tracked
        self.start_frames[tracks_pool] = frame_ids

        if frame_ids == 1:
            self.is_activated_tracks[tracks_pool] = True

        return tracks_pool

    def reactivate(self, track_pool: np.ndarray,
                   dets_storage: TrackStorage, dets_pool: np.ndarray,
                   frame_ids: int | np.ndarray, with_reid: bool):
        if track_pool.shape[0] == 0 or dets_pool.shape[0] == 0:
            return

        means, covs = list(zip(*[
            self.kalman_filter.update(
                self.means[track_pool][i],
                self.covs[track_pool][i],
                dets_storage.dets[dets_pool][i])
            for i in range(len(track_pool))]))

        self.means[track_pool] = np.array(means)
        self.covs[track_pool] = np.array(covs)

        self.frame_ids[track_pool] = frame_ids
        self.tracklet_lens[track_pool] = 0
        self.states[track_pool] = TrackState.Tracked
        self.is_activated_tracks[track_pool] = True

        # TODO: check if embeddings is being updated too
        # TODO: Check how embedding should be updated, is there any EMA?
        # if with_reid:
        #     self.embs[track_pool] = dets_storage.embs[dets_pool]

        self.confs[track_pool] = dets_storage.confs[dets_pool]
        self.classes[track_pool] = dets_storage.classes[dets_pool]
        self.det_ids[track_pool] = dets_storage.det_ids[dets_pool]

    def cleanup(self, save_pool):
        self._manager.cleanup(save_pool)

