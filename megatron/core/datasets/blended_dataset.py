# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
import logging
from math import floor
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import copy

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import normalize
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_VERBOSE = False


class BlendedDataset(torch.utils.data.Dataset):
    """Conjugating class for a set of MegatronDataset instances

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[Union[int, float]]): The weights that determine the dataset blend ratios

        size (Optional[int]): The number of samples to draw from the blend. If None, for each
            dataset index idx draw exactly weights[idx] samples from datasets[idx].

        config (BlendedMegatronDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: List[MegatronDataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        assert len(datasets) == len(weights)
        assert len(datasets) < 32767
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        assert all(map(lambda _: _.index_split == datasets[0].index_split, datasets))
        assert all(map(lambda _: _ > 0, weights))
        assert all(map(lambda _: type(_) == type(weights[0]), weights))
        if size is None and isinstance(weights[0], float):
            assert all(map(lambda _: _ == int(_), weights))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            log_single_rank(
                logger, logging.WARNING, f"Building a BlendedDataset for a single MegatronDataset"
            )

        if size is not None:
            weights = normalize(weights)

        self.datasets = datasets
        self.split = self.datasets[0].index_split
        self.weights = weights
        self.size = size
        self.config = config

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["split"] = self.split.name
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size

        self.unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        self.dataset_index, self.dataset_sample_index = self._build_indices()
        
        self.set_skip_config()
    
    def set_skip_config(self):
        from megatron.training import get_args
        args = get_args()
        #读取文件 和 env相关的配置，得到skip的数据；
        self.arsenal_skip_config = None
        # 读取auto_skip_steps_record.json中的文件
        skip_steps = {}
        auto_skip_file = os.getenv("auto_skip_file", './auto_skip_steps_record.json')
        skip_recoder = os.path.join(os.path.dirname(auto_skip_file), "do_not_delete_" + os.path.basename(auto_skip_file))
        auto_skip_interval = int(os.getenv("auto_skip_interval", "20"))
        auto_skip_threshold = int(os.getenv("auto_skip_threshold", "1"))
        manual_skip_config = os.getenv("manual_skip_config", "{}")
        max_skip_step = int(os.getenv("max_skip_step", "200"))

        if os.path.exists(auto_skip_file):
            with open(auto_skip_file, 'r') as f:
                config = f.read().strip()
                if len(config) > 0:
                    skip_steps = json.loads(config)
            # 这里不能做同步，因为不是所有的rank都会进入到这个blenddataset的构建过程中
            #torch.distributed.barrier()
        save_interval = args.save_interval
        next_ckpt_step = None
        if save_interval and save_interval > 0:
            next_ckpt_step = int((floor((args.iteration+1)/save_interval)+1) * save_interval)
        
        adapted_skip_steps = {}
        for step, count in skip_steps.items():
            step = int(step)
            count = int(count)
            if count >= auto_skip_threshold:
                interval = min(auto_skip_interval * (count-auto_skip_threshold+1), max_skip_step)
                left_step = max(step - interval/2, 0)
                adapted_skip_steps[left_step] = interval

        if len(adapted_skip_steps) > 0:
            log_single_rank(logger, logging.INFO, f"arsenal retrain - set auto skip config: {adapted_skip_steps}, next_ckpt_step: {next_ckpt_step}")
        else:
            log_single_rank(logger, logging.INFO, f"arsenal retrain - no auto skip config.")
        self.arsenal_skip_config = adapted_skip_steps
        
        manual_skip_config = manual_skip_config.strip()
        if len(manual_skip_config) > 0:
            manual_skip_config = json.loads(manual_skip_config)
        else:
            manual_skip_config = {}
        
        if len(manual_skip_config) > 0:
            log_single_rank(logger, logging.INFO, f"arsenal retrain - use manual skip config: {manual_skip_config}")
        else:
            log_single_rank(logger, logging.INFO, f"arsenal retrain - no manual skip config.")

        for step, interval in manual_skip_config.items():
            step = int(step)
            interval = int(interval)
            self.arsenal_skip_config[step] = interval

        self.gbs = int(args.global_batch_size)

        if torch.distributed.get_rank() == 0:
            with open(skip_recoder, "w+") as f:
                json.dump(self.arsenal_skip_config, f)
        
        if len(self.arsenal_skip_config) > 0:
            # 这里不能做同步，因为不是所有的rank都会进入到这个blenddataset的构建过程中
            # torch.distributed.barrier()
            # 所有rank都打印，方便对比不同rank上的差异
            logger.info(f"arsenal retrain - use auto skip config: {skip_steps} in {auto_skip_file}")
            logger.info(f"arsenal retrain - use skip gbs: {self.gbs}, final skip config: {self.arsenal_skip_config}")


    def __len__(self) -> int:
        if self.config.defer_npy_index_mmap:
            size = sum(self.weights)
            if self.size is not None:
                size = self.size
            return size

        return self.dataset_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        if self.dataset_index is None:
            self.dataset_index = numpy.load(
                self.path_to_dataset_index, allow_pickle=True, mmap_mode="r"
            )
            self.dataset_sample_index = numpy.load(
                self.path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r"
            )
        if self.arsenal_skip_config:
            tmp_idx = idx
            for start_step, skip_steps in self.arsenal_skip_config.items():
                if tmp_idx >= (int(start_step)-1)*self.gbs:
                    idx = idx + int(skip_steps)*self.gbs
            assert 0 <= idx < self.dataset_index.shape[0], f"idx: {idx} is out of range, dataset length: {self.dataset_index.shape[0]}"

        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        return {"dataset_id": dataset_id, **self.datasets[dataset_id][dataset_sample_id]}

    def _build_indices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The dataset index and the dataset sample index
        """
        if self.config.defer_npy_index_mmap:
            # NOTE(asolergi-nv): Direct path to lazy memmap the indexes
            get_path_to = lambda suffix: os.path.join(
                self.config.path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.split.name}-{suffix}",
            )
            self.path_to_dataset_index = get_path_to("dataset_index.npy")
            self.path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            return None, None

        path_to_cache = self.config.path_to_cache

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.split.name}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = (
                True
                if self.config.fast_cache_load
                else all(
                    map(
                        os.path.isfile,
                        [path_to_description, path_to_dataset_index, path_to_dataset_sample_index],
                    )
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):
            log_single_rank(
                logger, logging.INFO, f"Build and save the {type(self).__name__} indices"
            )

            # Build the dataset and dataset sample indexes
            log_single_rank(
                logger, logging.INFO, f"\tBuild and save the dataset and dataset sample indexes"
            )
            t_beg = time.time()
            from megatron.core.datasets import helpers

            if self.size is not None:
                dataset_index = numpy.zeros(self.size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(self.size, dtype=numpy.int64)
                helpers.build_blending_indices(
                    dataset_index,
                    dataset_sample_index,
                    self.weights,
                    len(self.datasets),
                    self.size,
                    _VERBOSE,
                )
            else:
                size = sum(self.weights)
                dataset_index = numpy.zeros(size, dtype=numpy.int16)
                dataset_sample_index = numpy.zeros(size, dtype=numpy.int64)
                helpers.build_exhaustive_blending_indices(
                    dataset_index, dataset_sample_index, self.weights, len(self.datasets)
                )

            dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
            for i, (_index, _size) in enumerate(zip(dataset_indices, dataset_sizes)):
                if len(self.datasets[_index]) < _size:
                    raise IndexError(
                        f"The {self.split.name} blend oversamples the contributing datasets and, "
                        f"for example, requests {_size} samples from "
                        f"{type(self.datasets[_index]).__name__} number {i} in excess of its size "
                        f"{len(self.datasets[_index])}. The current value of the config attribute "
                        f"mid_level_dataset_surplus may be increased, e.g. two- or ten-fold, from "
                        f"its current value ({self.config.mid_level_dataset_surplus}) to ensure a "
                        f"sufficient mid-level dataset sample margin from which to draw."
                    )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                # Save the indexes
                numpy.save(path_to_dataset_index, dataset_index, allow_pickle=True)
                numpy.save(path_to_dataset_sample_index, dataset_sample_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Cannot save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return dataset_index, dataset_sample_index

        log_single_rank(logger, logging.INFO, f"Load the {type(self).__name__} indices")

        log_single_rank(
            logger, logging.INFO, f"\tLoad the dataset index from {path_to_dataset_index}"
        )
        t_beg = time.time()
        dataset_index = numpy.load(path_to_dataset_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the dataset sample index from {path_to_dataset_sample_index}",
        )
        t_beg = time.time()
        dataset_sample_index = numpy.load(
            path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r"
        )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index
