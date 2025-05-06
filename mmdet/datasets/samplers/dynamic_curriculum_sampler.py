import math
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
from .multi_source_sampler_epoch import MultiSourceSamplerForEpoch


@DATA_SAMPLERS.register_module()
class DynamicCurriculumSampler(MultiSourceSamplerForEpoch):
    """Dynamic Curriculum Sampler that smoothly transitions from synthetic to real data.

    This sampler implements a curriculum learning strategy where the ratio of synthetic
    to real data smoothly changes over epochs using a cosine schedule.

    Args:
        dataset (Sized): A ConcatDataset with attribute `cumulative_sizes`.
        batch_size (int): Size of a mini-batch (per GPU).
        initial_synthetic_ratio (float): Initial ratio of synthetic data (default: 1.0).
        final_synthetic_ratio (float): Final ratio of synthetic data (default: 0.0).
        shuffle (bool): Whether to shuffle the datasets.
        seed (int, optional): Random seed. If None, a synchronized random seed
            will be used.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 initial_synthetic_ratio: float = 1.0,
                 final_synthetic_ratio: float = 0.0,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(dataset, batch_size, [1] * len(dataset.cumulative_sizes),
                        shuffle, seed)
        
        self.initial_synthetic_ratio = initial_synthetic_ratio
        self.final_synthetic_ratio = final_synthetic_ratio
        
        # Define relative ratios for synthetic and real datasets
        self.synthetic_ratios = [2, 1, 2, 1]  # Relative ratios between synthetic datasets
        self.real_ratios = [1, 1, 1]  # Equal ratios for real datasets
        
        # Validate dataset structure
        assert len(dataset.datasets) == len(self.synthetic_ratios) + len(self.real_ratios), \
            f'Number of datasets ({len(dataset.datasets)}) must match sum of synthetic ({len(self.synthetic_ratios)}) and real ({len(self.real_ratios)}) ratios'

    def _calculate_current_ratios(self, epoch: int) -> List[float]:
        """Calculate current sampling ratios based on epoch progress.

        Args:
            epoch (int): Current epoch number.

        Returns:
            List[float]: Current sampling ratios for all datasets.
        """
        # Calculate progress using cosine schedule
        progress = epoch / (self.max_epochs - 1)  # Normalized progress [0, 1]
        current_synthetic_ratio = self.final_synthetic_ratio + 0.5 * \
            (self.initial_synthetic_ratio - self.final_synthetic_ratio) * \
            (1 + math.cos(math.pi * progress))
        
        # Calculate synthetic dataset ratios
        synthetic_sum = sum(self.synthetic_ratios)
        synthetic_ratios = [r * current_synthetic_ratio / synthetic_sum 
                          for r in self.synthetic_ratios]
        
        # Calculate real dataset ratios
        real_sum = sum(self.real_ratios)
        real_ratios = [r * (1 - current_synthetic_ratio) / real_sum 
                      for r in self.real_ratios]
        
        # Combine ratios
        return synthetic_ratios + real_ratios

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch and update source ratio based on current epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.current_epoch = epoch
        if epoch < mixup_epochs:
            self.source_ratio = mixup_source_ratio
            self.aug_source_idx = 0
        else:
            self.source_ratio = real_source_ratio
            self.aug_source_idx = 4
        
        # Update num_per_source based on new source_ratio
        self.num_per_source = [
            int(self.batch_size * sr / sum(self.source_ratio)) 
            for sr in self.source_ratio
        ]
        self.num_per_source[0] = self.batch_size - sum(self.num_per_source[1:])
        
        # Reinitialize source indices
        self._init_source_indices()

    def __iter__(self) -> Iterator[int]:
        """Generate indices for the current epoch.

        Returns:
            Iterator[int]: Iterator of indices for the current epoch.
        """
        return super().__iter__()

    def __len__(self) -> int:
        """Get the length of the sampler.

        Returns:
            int: Length of the sampler.
        """
        return super().__len__()

# Example usage
mixup_epochs = 3
real_epochs = 2
max_epochs = mixup_epochs + real_epochs

# 采样比例
mixup_source_ratio = [4, 2, 4, 2, 1, 1, 1]  # synthetic + real
real_source_ratio = [0, 0, 0, 0, 1, 1, 1]   # only real 