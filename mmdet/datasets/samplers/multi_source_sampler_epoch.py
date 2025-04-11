# import itertools
# from typing import Iterator, List, Optional, Sized, Union

# import numpy as np
# import torch
# from mmengine.dataset import BaseDataset
# from mmengine.dist import get_dist_info, sync_random_seed
# from torch.utils.data import Sampler

# from mmdet.registry import DATA_SAMPLERS


# @DATA_SAMPLERS.register_module()
# class MultiSourceSamplerForEpoch(Sampler):
#     r"""Multi-Source Sampler for Epoch-based Distributed Training.

#     This sampler generates a full (global) list of indices based on the provided
#     source ratio and then slices them so that each process (GPU) only gets its
#     own share.

#     Args:
#         dataset (Sized): A ConcatDataset with attribute `cumulative_sizes`.
#         batch_size (int): Size of a mini-batch (per GPU).
#         source_ratio (list[int | float]): The sampling ratio of different
#             source datasets in a mini-batch.
#         shuffle (bool): Whether to shuffle the datasets.
#         seed (int, optional): Random seed. If None, a synchronized random seed
#             will be used.
#     """

#     def __init__(self,
#                  dataset: Sized,
#                  batch_size: int,
#                  source_ratio: List[Union[int, float]],
#                  shuffle: bool = True,
#                  seed: Optional[int] = None) -> None:

#         # Validate inputs
#         assert hasattr(dataset, 'cumulative_sizes'), (
#             f'The dataset must be a ConcatDataset with cumulative_sizes, but got {dataset}'
#         )
#         assert isinstance(batch_size, int) and batch_size > 0, (
#             f'batch_size must be a positive integer, but got {batch_size}'
#         )
#         assert isinstance(source_ratio, list), (
#             f'source_ratio must be a list, but got {source_ratio}'
#         )
#         assert len(source_ratio) == len(dataset.cumulative_sizes), (
#             f'The length of source_ratio must equal the number of datasets in the ConcatDataset, '
#             f'but got {source_ratio}'
#         )

#         # Distributed info: each process has its own rank and the total number of processes
#         self.rank, self.world_size = get_dist_info()

#         self.dataset = dataset
#         # Prepend 0 to cumulative_sizes to compute offsets for each dataset
#         self.cumulative_sizes = [0] + dataset.cumulative_sizes
#         self.batch_size = batch_size
#         self.source_ratio = source_ratio

#         # Determine how many samples per source in one batch.
#         # The first source is adjusted to ensure the total equals batch_size.
#         self.num_per_source = [
#             int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
#         ]
#         self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])
#         assert sum(self.num_per_source) == batch_size, (
#             f'The sum of num_per_source ({self.num_per_source}) must equal batch_size, '
#             f'got {sum(self.num_per_source)}'
#         )

#         # Use synchronized random seed if seed is not provided.
#         self.base_seed = sync_random_seed() if seed is None else seed
#         self.shuffle = shuffle

#         # For epoch-based shuffling.
#         self.current_epoch = 0
#         self.source_indices = dict()  # Holds the shuffled indices per source
#         self._init_source_indices()

#     def _init_source_indices(self) -> None:
#         """Initialize or reinitialize per-source indices based on the current epoch."""
#         self.source_indices = {}
#         for source, ds in enumerate(self.dataset.datasets):
#             sample_size = len(ds)
#             g = torch.Generator()
#             g.manual_seed(self.base_seed + self.current_epoch)
#             if self.shuffle:
#                 indices = torch.randperm(sample_size, generator=g).tolist()
#             else:
#                 indices = list(range(sample_size))
#             self.source_indices[source] = indices

#     def set_epoch(self, epoch: int) -> None:
#         """Set the epoch to update shuffling.

#         This is called by the epoch-based runner to ensure a new random order per epoch.
#         """
#         self.current_epoch = epoch
#         self._init_source_indices()

#     def __iter__(self) -> Iterator[int]:
#         # Compute the total number of samples (global) to yield in this epoch.
#         total_samples = len(self.dataset)
#         num_batches = total_samples // self.batch_size
#         total_yield = num_batches * self.batch_size  # Global sample count for the epoch

#         global_indices = []  # Will hold indices for the full (global) epoch
#         yielded = 0

#         # Generate global batches until total_yield is met.
#         while yielded < total_yield:
#             batch_buffer = []
#             # For each source dataset, select the required number of indices.
#             for source, num in enumerate(self.num_per_source):
#                 indices_for_source = []
#                 while len(indices_for_source) < num:
#                     # If insufficient indices remain, reshuffle for a new pass.
#                     if not self.source_indices[source]:
#                         sample_size = len(self.dataset.datasets[source])
#                         g = torch.Generator()
#                         g.manual_seed(self.base_seed + self.current_epoch + 1)
#                         if self.shuffle:
#                             self.source_indices[source] = torch.randperm(sample_size, generator=g).tolist()
#                         else:
#                             self.source_indices[source] = list(range(sample_size))
#                     needed = num - len(indices_for_source)
#                     take = self.source_indices[source][:needed]
#                     indices_for_source.extend(take)
#                     # Remove the taken indices.
#                     self.source_indices[source] = self.source_indices[source][needed:]
#                 # Adjust indices for the concatenated dataset.
#                 batch_buffer.extend([idx + self.cumulative_sizes[source] for idx in indices_for_source])
#             # Optionally shuffle the indices within the batch.
#             if self.shuffle:
#                 batch_buffer = np.random.permutation(batch_buffer).tolist()
#             global_indices.extend(batch_buffer)
#             yielded += self.batch_size

#         # Ensure the global indices list is the desired length.
#         global_indices = global_indices[:total_yield]

#         # --- Distributed Slice ---
#         # Slice the global indices to yield only the indices assigned to this process.
#         local_indices = global_indices[self.rank:total_yield:self.world_size]
#         for idx in local_indices:
#             yield idx

#     def __len__(self) -> int:
#         # Global epoch length.
#         total_samples = len(self.dataset)
#         num_batches = total_samples // self.batch_size
#         total_yield = num_batches * self.batch_size
#         # Each process (GPU) gets an approximately equal share.
#         return total_yield // self.world_size

import itertools
import math
from typing import Iterator, List, Optional, Sized, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class MultiSourceSamplerForEpoch(Sampler):
    r"""Multi-Source Sampler for Epoch-based Distributed Training.

    This sampler generates a full (global) list of indices based on the provided
    source ratio and then slices them so that each process (GPU) only gets its
    own share.

    An extra parameter `aug_source_idx` is added. When provided, it indicates
    that the dataset with index `aug_source_idx` is to be iterated exactly once
    in an epoch (i.e. all its samples are used only one time), and the other
    datasets will be sampled to supplement the mini batch according to the
    specified source ratio.

    Args:
        dataset (Sized): A ConcatDataset with attribute `cumulative_sizes`.
        batch_size (int): Size of a mini-batch (per GPU).
        source_ratio (list[int | float]): The sampling ratio of different
            source datasets in a mini-batch.
        shuffle (bool): Whether to shuffle the datasets.
        seed (int, optional): Random seed. If None, a synchronized random seed
            will be used.
        aug_source_idx (int, optional): The index of the dataset that is to be
            used as the anchor and iterated exactly once in an epoch. If None,
            no anchor dataset is enforced.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 source_ratio: List[Union[int, float]],
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 aug_source_idx: Optional[int] = None) -> None:

        # Validate inputs
        assert hasattr(dataset, 'cumulative_sizes'), (
            f'The dataset must be a ConcatDataset with cumulative_sizes, but got {dataset}'
        )
        assert isinstance(batch_size, int) and batch_size > 0, (
            f'batch_size must be a positive integer, but got {batch_size}'
        )
        assert isinstance(source_ratio, list), (
            f'source_ratio must be a list, but got {source_ratio}'
        )
        assert len(source_ratio) == len(dataset.cumulative_sizes), (
            f'The length of source_ratio must equal the number of datasets in the ConcatDataset, '
            f'but got {source_ratio}'
        )
        if aug_source_idx is not None:
            assert 0 <= aug_source_idx < len(dataset.datasets), (
                f'aug_source_idx must be between 0 and {len(dataset.datasets)-1}, but got {aug_source_idx}'
            )

        # Distributed info: each process has its own rank and the total number of processes
        self.rank, self.world_size = get_dist_info()

        self.dataset = dataset
        # Prepend 0 to cumulative_sizes to compute offsets for each dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.aug_source_idx = aug_source_idx

        # Determine how many samples per source in one batch.
        # The first source is adjusted to ensure the total equals batch_size.
        self.num_per_source = [
            int(batch_size * sr / sum(source_ratio)) for sr in source_ratio
        ]
        self.num_per_source[0] = batch_size - sum(self.num_per_source[1:])
        assert sum(self.num_per_source) == batch_size, (
            f'The sum of num_per_source ({self.num_per_source}) must equal batch_size, '
            f'got {sum(self.num_per_source)}'
        )

        # Use synchronized random seed if seed is not provided.
        self.base_seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle

        # For epoch-based shuffling.
        self.current_epoch = 0
        self.source_indices = dict()  # Holds the shuffled indices per source
        self._init_source_indices()

    def _init_source_indices(self) -> None:
        """Initialize or reinitialize per-source indices based on the current epoch."""
        self.source_indices = {}
        for source, ds in enumerate(self.dataset.datasets):
            sample_size = len(ds)
            g = torch.Generator()
            g.manual_seed(self.base_seed + self.current_epoch)
            if self.shuffle:
                indices = torch.randperm(sample_size, generator=g).tolist()
            else:
                indices = list(range(sample_size))
            self.source_indices[source] = indices

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch to update shuffling.

        This is called by the epoch-based runner to ensure a new random order per epoch.
        """
        self.current_epoch = epoch
        self._init_source_indices()

    def __iter__(self) -> Iterator[int]:
        # 如果设置了 anchor 数据源，则以该数据源决定 epoch 的完整批次数
        if self.aug_source_idx is not None:
            anchor_total = len(self.dataset.datasets[self.aug_source_idx])
            n_anchor = self.num_per_source[self.aug_source_idx]
            num_batches = anchor_total // n_anchor
        else:
            total_samples = len(self.dataset)
            num_batches = total_samples // self.batch_size

        total_yield = num_batches * self.batch_size  # Global sample count for the epoch

        global_indices = []  # Will hold indices for the full (global) epoch

        # 如果设置了 anchor，则提前截取该数据源的所有索引（正好用一次，不重复）
        if self.aug_source_idx is not None:
            anchor_indices_all = self.source_indices[self.aug_source_idx]
            # 截取刚好 num_batches * n_anchor 的索引，不足的部分直接丢弃
            anchor_indices = anchor_indices_all[:num_batches * self.num_per_source[self.aug_source_idx]]
            # 更新该数据源的索引，保证不会重复使用
            self.source_indices[self.aug_source_idx] = anchor_indices[num_batches * self.num_per_source[self.aug_source_idx]:]
        # 开始生成每个 batch 的全局索引
        yielded = 0
        while yielded < total_yield:
            batch_buffer = []
            for source, num in enumerate(self.num_per_source):
                indices_for_source = []
                if self.aug_source_idx is not None and source == self.aug_source_idx:
                    # 对于 anchor 数据源，从提前截取的列表中取出，不允许重复采样
                    start = yielded // self.batch_size * num
                    end = start + num
                    indices_for_source = self.source_indices[source][:num]  # 或者直接从 anchor_indices 切片
                    # 为了安全起见，可以提前计算好 anchor_indices 列表，再按批次取出:
                    if len(anchor_indices) < (yielded // self.batch_size + 1) * num:
                        raise ValueError("Not enough anchor indices for a full batch.")
                    indices_for_source = anchor_indices[(yielded // self.batch_size) * num: (yielded // self.batch_size + 1) * num]
                else:
                    # 对于其它数据源，逻辑保持不变：若不足则重洗
                    while len(indices_for_source) < num:
                        if not self.source_indices[source]:
                            sample_size = len(self.dataset.datasets[source])
                            g = torch.Generator()
                            g.manual_seed(self.base_seed + self.current_epoch + 1)
                            if self.shuffle:
                                self.source_indices[source] = torch.randperm(sample_size, generator=g).tolist()
                            else:
                                self.source_indices[source] = list(range(sample_size))
                        needed = num - len(indices_for_source)
                        take = self.source_indices[source][:needed]
                        indices_for_source.extend(take)
                        # Remove the taken indices.
                        self.source_indices[source] = self.source_indices[source][needed:]
                # Adjust indices for the concatenated dataset.
                batch_buffer.extend([idx + self.cumulative_sizes[source] for idx in indices_for_source])
            if self.shuffle:
                batch_buffer = np.random.permutation(batch_buffer).tolist()
            global_indices.extend(batch_buffer)
            yielded += self.batch_size

        global_indices = global_indices[:total_yield]

        # --- Distributed Slice ---
        local_indices = global_indices[self.rank:total_yield:self.world_size]
        for idx in local_indices:
            yield idx

    def __len__(self) -> int:
        if self.aug_source_idx is not None:
            anchor_total = len(self.dataset.datasets[self.aug_source_idx])
            n_anchor = self.num_per_source[self.aug_source_idx]
            num_batches = anchor_total // n_anchor
        else:
            total_samples = len(self.dataset)
            num_batches = total_samples // self.batch_size
        total_yield = num_batches * self.batch_size
        return total_yield // self.world_size