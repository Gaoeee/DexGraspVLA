import zarr
import numpy as np
from controller.common.replay_buffer import ReplayBuffer

class ZarrImageReference:
    """Lazy loading image data reference"""
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape  # right_cam_img：(3825, 480, 640, 3) rgbm：(3825, 480, 640, 4)
        self.dtype = zarr_array.dtype  # uint8 
        self.chunks = zarr_array.chunks # (3, 480, 640, 3) 三个元素划为一个 chunk rgbm：(2, 480, 640, 4)

    def __getitem__(self, idx):
        return self.zarr_array[idx]

    def __len__(self):
        return len(self.zarr_array)

class StreamingReplayBuffer(ReplayBuffer):
    """
    Streaming replay buffer, mainly used for handling large-scale data
    """
    def __init__(self, root=None):
        self._data = None
        self._meta = None
        self._episode_ends = None
        self.zarr_path = None
        
    @classmethod
    def copy_from_path(cls, path, keys=None):
        """Create StreamingReplayBuffer from zarr file"""
        buffer = cls()
        buffer.zarr_path = path
        
        # Open zarr storage
        src = zarr.open(path, mode='r')
        
        # Load metadata
        buffer._meta = dict()
        if 'meta' in src:
            for key, value in src['meta'].items():  # 'episode_ends'
                if len(value.shape) == 0:  #标量数据
                    buffer._meta[key] = np.array(value)  # 转成矩阵
                else:
                    buffer._meta[key] = value[:]
        
        # Set data references
        buffer._data = dict()
        if 'data' not in src:
            src_data = src
        else:
            src_data = src['data']
            
        if keys is None:
            keys = list(src_data.keys())
            
        for key in keys:
            arr = src_data[key]
            # Use references for image data, load other data completely
            if key in ['rgbm', 'right_cam_img']:
                buffer._data[key] = ZarrImageReference(arr)
            else:
                buffer._data[key] = arr[:]  # right_state(3825, 13) action(3825, 13)
                
        return buffer

    @property
    def data(self):
        """Get data dictionary"""
        return self._data

    @property
    def meta(self):
        """Get metadata dictionary"""
        return self._meta

    @property
    def episode_ends(self):
        """Get episode ending indices"""
        if self._episode_ends is None:
            if 'episode_ends' in self.meta:
                self._episode_ends = self.meta['episode_ends']
            else:
                # Get length of first data
                first_data = next(iter(self.data.values()))
                self._episode_ends = np.array([len(first_data)])
        return self._episode_ends

    def get_episode(self, idx, copy=False):
        """Get data for specified episode"""
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        
        result = dict()
        for key, value in self.data.items():
            x = value[start_idx:end_idx]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """Get data for specified step range"""
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def __len__(self):
        """Get data length"""
        if len(self.episode_ends) == 0:
            return 0
        return int(self.episode_ends[-1])

    def __getitem__(self, key):
        """Get data for specified key"""
        return self.data[key]

    def keys(self):
        """Get all data keys"""
        return self.data.keys()

    def values(self):
        """Get all data values"""
        return self.data.values()

    def items(self):
        """Get all data items"""
        return self.data.items()

    def __contains__(self, key):
        """Check if contains specified key"""
        return key in self.data

    @property
    def n_episodes(self):
        """Get total number of episodes"""
        return len(self.episode_ends)

    def get_episode_lengths(self):
        """Get length of each episode"""
        ends = self.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        return lengths