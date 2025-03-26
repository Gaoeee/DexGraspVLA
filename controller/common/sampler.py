from typing import Optional
import numpy as np
import numba
from controller.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int,  #64
    episode_mask: np.ndarray,
    pad_before: int=0, # 在序列开头允许的最大填充步数 0
    pad_after: int=0, # 在序列结尾允许的最大填充步数 63
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):  # len=51
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx  # 75  # 当前 episode 的长度
        
        min_start = -pad_before  # 0
        max_start = episode_length - sequence_length + pad_after #75-64+63=74
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx  # 75个一组 0开始顺序叠加
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx  #[64~75]+start_idx
            start_offset = buffer_start_idx - (idx+start_idx)  # 0
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx  # idx<11：0，idx>=11:idx-11
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,  # 缓冲区中的起始和结束索引  buffer_end_idx75个一组 75x封顶 75应该是一个clip的长度
                sample_start_idx, sample_end_idx]) # 输出样本中的起始和结束索引 （考虑填充） sample_start_idx-sample_end_idx最大64，不到64buffer_end_idx-sample_start_idx，后续可以用0padding至64
    """
    indices:
        [[ 0, 64,  0, 64],
        [1, 65, 0, 64],
        [2, 66, 0, 64],
        ...
        [10, 74, 0, 64],
        [11, 75, 0, 64],
        [12, 75, 0, 63],
        ...
        [74, 75, 0, 1],
        [75, 139, 0, 64],
        ...
        [148,150,0,1],
        [149,214,0,64],
        ...
        [3824, 3825, 0, 1]
    """
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool) # false组成的array
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,  #64
        pad_before:int=0, # 0
        pad_after:int=0, #63
        keys=None,
        key_first_k=dict(), #'right_cam_img': 1,'rgbm':1
        episode_mask: Optional[np.ndarray]=None,  # train_mask
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:] #shape 51
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask): #至少存在一个true
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                ) # (3825,4)
        else:
            indices = np.zeros((0,4), dtype=np.int64)  # (0,4)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]  # idx=1463 (1463, 1500, 0, 37)
        result = dict()
        for key in self.keys:  # 4种数据
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k: # 非图片
                sample = input_arr[buffer_start_idx:buffer_end_idx]  # 状态或action直接用buffer填充
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)   #1 只放一帧观测图片
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)  # （n_data，480，640，3）
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data] #第一维的前k个即只放k帧观测图片
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):  # padding到sequence_length
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype) # (64, 480, 640, 3) (64,13)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]  # 不到sequence_length的用最后一个值填充
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
