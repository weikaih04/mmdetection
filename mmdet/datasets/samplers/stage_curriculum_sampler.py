from typing import Iterator, List, Optional, Sized, Union
import math

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
from .multi_source_sampler_epoch import MultiSourceSamplerForEpoch


@DATA_SAMPLERS.register_module()
class StageCurriculumSampler(MultiSourceSamplerForEpoch):
    """Stage-based Curriculum Sampler for two-stage training.

    This sampler implements a two-stage curriculum learning strategy:
    1. Mixup stage: Mix synthetic and real data with specified ratios, smoothly transitioning
       from initial_mixup_ratio to final_mixup_ratio using cosine schedule
    2. Real stage: Use only real data

    Args:
        dataset (Sized): A ConcatDataset with attribute `cumulative_sizes`.
        batch_size (int): Size of a mini-batch (per GPU).
        mixup_epochs (int): Number of epochs for mixup stage.
        real_epochs (int): Number of epochs for real data stage.
        initial_mixup_ratio (List[Union[int, float]]): Initial sampling ratios for mixup stage.
        final_mixup_ratio (List[Union[int, float]]): Final sampling ratios for mixup stage.
        real_source_ratio (List[Union[int, float]]): Sampling ratios for real stage.
        aug_source_mixup (int): Index of the anchor dataset for mixup stage.
        aug_source_real (int): Index of the anchor dataset for real stage.
        shuffle (bool): Whether to shuffle the datasets.
        seed (int, optional): Random seed. If None, a synchronized random seed
            will be used.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 mixup_epochs: int,
                 real_epochs: int,
                 initial_mixup_ratio: List[Union[int, float]],
                 final_mixup_ratio: List[Union[int, float]],
                 real_source_ratio: List[Union[int, float]],
                 aug_source_mixup: int = 0,
                 aug_source_real: int = 4,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        super().__init__(dataset, batch_size, initial_mixup_ratio, shuffle, seed)
        
        self.mixup_epochs = mixup_epochs
        self.real_epochs = real_epochs
        self.initial_mixup_ratio = initial_mixup_ratio
        self.final_mixup_ratio = final_mixup_ratio
        self.real_source_ratio = real_source_ratio
        self.aug_source_mixup = aug_source_mixup
        self.aug_source_real = aug_source_real
        
        # Validate dataset structure
        assert len(dataset.datasets) == len(initial_mixup_ratio) == len(final_mixup_ratio) == len(real_source_ratio), \
            f'Number of datasets ({len(dataset.datasets)}) must match length of all ratio lists'
        
        # Validate anchor indices
        assert 0 <= aug_source_mixup < len(dataset.datasets), \
            f'aug_source_mixup must be between 0 and {len(dataset.datasets)-1}, but got {aug_source_mixup}'
        assert 0 <= aug_source_real < len(dataset.datasets), \
            f'aug_source_real must be between 0 and {len(dataset.datasets)-1}, but got {aug_source_real}'

    def _calculate_current_mixup_ratio(self, epoch: int) -> List[float]:
        """Calculate current mixup ratios based on epoch progress using cosine schedule.

        Args:
            epoch (int): Current epoch number.

        Returns:
            List[float]: Current sampling ratios for mixup stage.
        """
        # Calculate progress using cosine schedule
        progress = epoch / (self.mixup_epochs - 1)  # Normalized progress [0, 1]
        current_ratios = []
        
        for init_ratio, final_ratio in zip(self.initial_mixup_ratio, self.final_mixup_ratio):
            # Smoothly transition from initial to final ratio using cosine schedule
            current_ratio = final_ratio + 0.5 * (init_ratio - final_ratio) * (1 + math.cos(math.pi * progress))
            current_ratios.append(current_ratio)
            
        return current_ratios

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch and update source ratio based on current stage.

        Args:
            epoch (int): Current epoch number.
        """
        self.current_epoch = epoch
        
        # Determine current stage and update source ratio
        if epoch < self.mixup_epochs:
            self.source_ratio = self._calculate_current_mixup_ratio(epoch)
            self.aug_source_idx = self.aug_source_mixup
        else:
            self.source_ratio = self.real_source_ratio
            self.aug_source_idx = self.aug_source_real
        
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