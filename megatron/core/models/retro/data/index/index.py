# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import abc
import os
import numpy as np
import torch
from typing import Any, List, Tuple

from megatron.core.models.retro.data.config import Embedder, RetroPreprocessingConfig
from megatron.core.models.retro.data.external_libs import faiss
from megatron.core.models.retro.data.utils import GPTToTextDataset

from .utils import get_index_dir


class Index(abc.ABC):

    '''Abstract base class for indexes.

    *Note* : While currently only Faiss-based classes are implemented, in the
    future, this class will be extended with other types of indexes that have
    different performance-accuracy trade-offs.

    The primary methods to override are:
    - train() : Train index on the sampled training chunks.
    - add() : Add all training chunks to index.
    '''

    @classmethod
    def c_verbose(cls, index: faiss.Index, v: bool) -> None:
        '''Make index object verbose.'''
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v)

    def get_empty_index_path(self, config: RetroPreprocessingConfig) -> str:
        return os.path.join(
            get_index_dir(config), "empty_%.3f.faissindex" % config.retro_index_train_load_fraction,
        )

    def get_empty_index(self, config: RetroPreprocessingConfig) -> faiss.Index:
        return faiss.read_index(self.get_empty_index_path(config))

    def get_added_index_path(self, config: RetroPreprocessingConfig) -> str:
        return os.path.join(
            get_index_dir(config),
            "added_%.3f_%.3f.faissindex"
            % (config.retro_index_train_load_fraction, config.retro_index_add_load_fraction,),
        )

    def get_added_index(self, config: RetroPreprocessingConfig) -> faiss.Index:
        return faiss.read_index(self.get_added_index_path(config))

    @abc.abstractmethod
    def train(self, *args: List[Any]) -> None:
        pass

    @abc.abstractmethod
    def add(self, *args: List[Any]) -> str:
        pass

    def embed_text_dataset_block(self, embedder: Embedder, text_dataset: GPTToTextDataset, _range: Tuple) -> np.ndarray:
        '''Embed a range of a text dataset.'''
        sub_dataset = torch.utils.data.Subset(text_dataset, range(*_range))
        return embedder.embed_text_dataset(sub_dataset)
