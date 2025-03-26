from typing import Dict
import torch
import numpy as np
import copy
from controller.common.pytorch_util import dict_apply
from controller.common.streaming_replay_buffer import StreamingReplayBuffer
from controller.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from controller.model.common.normalizer import LinearNormalizer
from controller.dataset.base_dataset import BaseImageDataset
import torch.nn.functional as F

class MaskImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(518, 518)
            ):
        
        super().__init__()
        self.image_size = image_size
        
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=['right_cam_img', 'rgbm', 'right_state', 'action'])
            self.replay_buffers.append(replay_buffer)
            
            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,  # len(self.episode_ends)
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)
            self.train_masks.append(train_mask)
            
            # Create sampler 重要
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon, # 64
                pad_before=pad_before, # 0
                pad_after=pad_after, # 63
                episode_mask=train_mask,
                key_first_k=dict(right_cam_img=n_obs_steps, rgbm=n_obs_steps))
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                key_first_k=dict(right_cam_img=self.n_obs_steps, rgbm=self.n_obs_steps))
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set

    def _process_mask_image_batch(self, images):
        """Process images in batch"""
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        mask = torch.from_numpy(images[..., 3:]).float() # [T, H, W, 1]
        
        # Process RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,  # (518, 518)
            mode='bilinear',
            align_corners=False
        )

        # Process mask
        mask = mask.permute(0, 3, 1, 2)  # [T, 1, H, W]
        mask = F.interpolate(
            mask,
            size=self.image_size,
            mode='nearest'
        )

        mask = (mask > 0.5).float()

        # Combine
        combined = torch.cat([rgb, mask], dim=1)  # [T, 4, H, W]
        return combined.numpy()

    def _process_image_batch(self, images):
        """Process images in batch"""
        rgb = torch.from_numpy(images[..., :3]).float()  # [T, H, W, 3]
        
        # Process RGB
        rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]

        rgb = F.interpolate(
            rgb / 255.0,
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        return rgb.numpy()
    
    def _sample_to_data(self, sample):
        right_state = sample['right_state'].astype(np.float32) #（64，13）
        T_slice = slice(self.n_obs_steps)  # slice(None, 1, None) slice(start, stop, step) 用于定义切片的范围和步长

        # Process all images in batch
        mask_processed_frames = self._process_mask_image_batch(sample['rgbm'][T_slice])  # 取第一张 插值为518 518
        processed_frames = self._process_image_batch(sample['right_cam_img'][T_slice])

        data = {
            'obs': {
                'rgbm': mask_processed_frames,
                'right_cam_img': processed_frames,
                'right_state': right_state[T_slice]  # 只取1帧
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        # Merge all data
        actions = []
        right_states = []
        for rb in self.replay_buffers:
            actions.append(rb['action'])
            right_states.append(rb['right_state'])
            
        data = {
            'action': np.concatenate(actions, axis=0),
            'right_state': np.concatenate(right_states, axis=0)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):  # 遍历所有数据集？
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)  # 跳转函数取数据 一个sequence64个数据
                break
            curr_idx -= length
            
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy) # 递归处理字典将numpy转tensor
        return torch_data

    def __len__(self):
        return sum(self.sampler_lens)