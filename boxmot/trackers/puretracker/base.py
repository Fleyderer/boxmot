import numpy as np

class TrackManager:
    """
    Class to manage storage for tracks
    """
    def __init__(self, size: int):
        # Dictionary to map track IDs to storage positions
        self.track_to_storage = {}  
        # Set to keep track of free storage positions
        self.free_indices = set(range(size))  

    def add(self, track_id: int):
        if track_id in self.track_to_storage:
            raise ValueError(f"Track ID {track_id} already exists.")

        if not self.free_indices:
            raise ValueError("No free storage positions available.")

        # Find a free index
        index = self.free_indices.pop()
        self.track_to_storage[track_id] = index

    def remove(self, track_id: int):
        if track_id not in self.track_to_storage:
            raise ValueError(f"Track ID {track_id} does not exist.")

        # Get the index of the track to be removed
        index = self.track_to_storage.pop(track_id)
        self.free_indices.add(index)

    def get_ids(self, track_ids: list[int]) -> list[int]:
        return [self.track_to_storage[track_id] for track_id in track_ids]
    

class TrackStorage:

    def __init__(self):
        self.confs = None
        self.classes = None
        self.det_ids = None
        self.max_obs = None
        self.embs = None
        self.kalman_filters = None
        self.means = None
        self.covs = None
        self.is_activated_flags = None
        self.tracklet_lens = None
        self.history_observations = None
