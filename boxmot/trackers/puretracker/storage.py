from __future__ import annotations
from typing import List, Any, Union
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.utils.ops import xyxy2xywh, xywh2xyxy
from boxmot.trackers.puretracker.pool import ids_from_pool
import numpy as np


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
        return self.storage._data_dict[self.property_name].__getattr__(name)

    def __getitem__(self, track_ids: Union[int, List[int]]
                    ) -> Union[Any, np.ndarray]:
        st_indices = self.storage.translate_track_ids(track_ids)
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
                (self.storage._size, *values.shape), dtype=values.dtype)

        self.storage._data_dict[self.property_name][st_indices] = values

    def __eq__(self, value):
        track_ids, storage_ids = list(
            zip(*self.storage._manager._track_to_storage.items()))

        data = self.storage._data_dict[self.property_name][list(storage_ids)]

        return np.array(track_ids)[data == value]
    
    def __lt__(self, value):
        track_ids, storage_ids = list(
            zip(*self.storage._manager._track_to_storage.items()))

        data = self.storage._data_dict[self.property_name][list(storage_ids)]

        return np.array(track_ids)[data < value]

    def __gt__(self, value):    
        track_ids, storage_ids = list(
            zip(*self.storage._manager._track_to_storage.items()))

        data = self.storage._data_dict[self.property_name][list(storage_ids)]

        return np.array(track_ids)[data > value]
    
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
    kalman_filter = KalmanFilterXYWH()
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

    def __init__(self, size: int):
        self._data_dict = {}
        self._size = size
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

        if len(track_ids) > self._manager.space_left():
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

        multi_mean[self.states[pool] != TrackState.Tracked, 7] = 0

        multi_mean, multi_covariance = self.kalman_filter.multi_predict(
            multi_mean, multi_covariance)

        self.means[pool] = multi_mean
        self.covs[pool] = multi_covariance

    def set(self, track_ids: np.ndarray,
            dets: np.ndarray, embs: np.ndarray = None):
        self.dets[track_ids] = xyxy2xywh(dets[:, 0:4])
        self.confs[track_ids] = dets[:, 4]
        self.classes[track_ids] = dets[:, 5]
        self.det_ids[track_ids] = dets[:, 6]
        if embs is not None:
            self.embs[track_ids] = embs
        self.means[track_ids] = np.nan
        self.covs[track_ids] = np.nan
        self.states[track_ids] = TrackState.New
        self.is_activated_tracks[track_ids] = False
        self.tracklet_lens[track_ids] = 0
        self.frame_ids[track_ids] = 0

    def update(self, track_pool: np.ndarray,
               dets_storage: TrackStorage, dets_pool: np.ndarray,
               frame_ids: int | np.ndarray, with_reid: bool):
        self.frame_ids[track_pool] = frame_ids
        self.tracklet_lens[track_pool] += 1
        means, covs = list(zip(*[
            self.kalman_filter.update(
                self.means[track_pool][i],
                self.covs[track_pool][i],
                dets_storage[dets_pool][i])
            for i in range(len(track_pool))]))

        self.means[track_pool] = np.array(means)
        self.covs[track_pool] = np.array(covs)

        if with_reid:
            self.embs[track_pool] = dets_storage.embs[dets_pool]

        self.confs[track_pool] = dets_storage.confs[dets_pool]
        self.classes[track_pool] = dets_storage.classes[dets_pool]
        self.det_ids[track_pool] = dets_storage.det_ids[dets_pool]

    def activate(self, dets_storage: TrackStorage, dets_pool: np.ndarray,
                 frame_ids: int | np.ndarray, with_reid: bool):
        """Start a new tracklet"""
        start_id = self._manager.get_max_id() + 1
        tracks_pool = range(start_id, start_id + dets_pool.sum())

        self.set(tracks_pool, 
                 dets_storage.dets[dets_pool], 
                 dets_storage.embs[dets_pool] if with_reid else None)

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

    def reactivate(self, track_pool: np.ndarray,
                   dets_storage: TrackStorage, dets_pool: np.ndarray,
                   frame_ids: int | np.ndarray, with_reid: bool):

        means, covs = list(zip(*[
            self.kalman_filter.update(
                self.means[track_pool][i],
                self.covs[track_pool][i],
                dets_storage[dets_pool][i])
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
