# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
from typing import Literal
from collections import OrderedDict

import numpy as np


class ClassStorage:

    def __init__(self,
                 cls_value: int,
                 conf: float,
                 mode: Literal["vote", "last"] = "last",
                 max_len: int = 10):
        
        self._mode = mode

        if self._mode == "vote":
            self._cls_history: list[tuple[int, float]] = [(cls_value, conf)]
            self._cls_storage: dict = {cls_value: conf}
            self._max_len = max_len

        self._cls_value = cls_value

    def update(self, cls_value: int, conf: float):

        if self._mode == "vote":

            self._cls_history.append((cls_value, conf))
            self._cls_storage[cls_value] += conf

            if len(self._cls_history) > self._max_len:
                rm_cls_value, rm_cls_conf = self._cls_history[0]
                self._cls_storage[rm_cls_value] -= rm_cls_conf
                self._cls_history = self._cls_history[1:]

            self._cls_value = max(self._cls_storage.items(),
                                  key=lambda x: x[1])[0]

        elif self._mode == "last":
            self._cls_value = cls_value

    def get(self):
        return self.__int__()

    def __int__(self):
        return int(self._cls_value)


class EmbeddingStorage:
    _mode = None
    _ema_alpha = None
    _max_len = None

    def __init__(self, emb: np.ndarray):

        self._emb_history = []
        self._emb = emb / np.linalg.norm(emb)
        self._last_emb = self._emb

    def _update_emb_history(self, emb: np.ndarray):
        self._emb_history.append(emb)
        if len(self._emb_history) > self._max_len:
            self._emb_history = self._emb_history[1:]

    def update(self, emb: np.ndarray):
        emb /= np.linalg.norm(emb)

        if self._mode == "ema":
            self._emb = (self._ema_alpha * self._emb +
                         (1 - self._ema_alpha) * emb)
        elif self._mode == "mean":
            self._update_emb_history(emb)
            self._emb = np.mean(self._emb_history, axis=0)
        elif self._mode == "last":
            self._emb = emb

        self._last_emb = emb

    def get(self):
        return self._emb

    def get_last(self):
        return self._last_emb


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()

    conf = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
