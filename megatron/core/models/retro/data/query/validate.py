# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

? ? ?

# import numpy as np
# import os
# import psutil
# import time
# import torch
# from tqdm import tqdm

# from megatron.core.models.retro.data.db.utils import \
#     get_merged_train_dataset as get_db_merged_train_dataset
# from megatron.core.models.retro.data.external_libs import faiss, h5py
# from megatron.core.models.retro.data.index.factory import IndexFactory
# from megatron.core.models.retro.data.index.utils import get_index_dir
# from megatron.core.models.retro.data.utils import (
#     get_blocks_by_rank,
#     GPTToTextDataset,
#     print_rank_0,
# )

# from .chunk_dataset import get_chunk_dataset_map as get_query_dataset_map


# def get_index(config, ondisk=False):
#     '''Read index from disk.'''

#     # Load index.
#     index_wrapper = IndexFactory.get_index(config.retro_index_type)
#     index_dir = get_index_dir(config)
#     added_index_path = index_wrapper.get_added_index_path(config)
#     if ondisk:
#         index = faiss.read_index(added_index_path, faiss.IO_FLAG_MMAP)
#     else:
#         index = faiss.read_index(added_index_path)

#     # Search parameters.
#     faiss.ParameterSpace().set_index_parameter(index, "efSearch",
#                                                config.retro_query_ef_search)
#     faiss.ParameterSpace().set_index_parameter(index, "nprobe",
#                                                config.retro_query_nprobe)

#     return index


# def embed_block(config, gpt_dataset, block):
#     '''Embed block of chunks.'''
#     text_block_dataset = torch.utils.data.Subset(
#         GPTToTextDataset(gpt_dataset, config.retro_tokenizers.gpt),
#         range(*block["range"]),
#     )
#     return config.retro_bert_embedders.mem.embed_text_dataset(text_block_dataset)


# def query_embeddings(config, db_dataset, index,
#                      embeddings, chunk_id_range,
#                      sample_map, n_chunks_per_sample,
#                      verbose=True):
#     '''Query neighbors of a block of embeddings.'''

#     # Query neighbor ids.
#     if verbose: print_rank_0("search.")
#     t = time.time()
#     assert index.ntotal > 0, "check we don't accidentally have an empty index."
#     _, query_neighbor_ids = \
#         index.search(embeddings, config.retro_query_num_neighbors_query)
#     if verbose: print_rank_0("  time : %.3f sec." % (time.time() - t))

#     # Filter banned neighbor ids.
#     if verbose: print_rank_0("filter banned neighbor ids.")
#     filtered_neighbor_ids = np.full(
#         shape=(len(query_neighbor_ids), config.retro_query_num_neighbors_save),
#         fill_value=-1,
#         dtype="int64",
#     )
#     min_chunk_id, max_chunk_id = chunk_id_range
#     for chunk_id in range(min_chunk_id, max_chunk_id):

#         sample_id = chunk_id // n_chunks_per_sample
#         sample = sample_map[sample_id]
#         sample_dataset_idx = sample["dataset_idx"].item()
#         sample_doc_ids = sample["doc_ids"].tolist()
#         sample_doc_tuples = [(sample_dataset_idx, d) for d in sample_doc_ids]
        
#         # Get valid neighbors (!= -1).
#         query_row = [ i for i in query_neighbor_ids[chunk_id-min_chunk_id]
#                       if i >= 0 ]

#         # Filter row.
#         filtered_row = [ i for i in query_row
#                          if tuple(db_dataset.doc_tuples[i].tolist())
#                          not in sample_doc_tuples ]
#         filtered_row = filtered_row[:config.retro_query_num_neighbors_save]
#         filtered_row += \
#             [-1] * (config.retro_query_num_neighbors_save - len(filtered_row))
#         filtered_neighbor_ids[chunk_id-min_chunk_id] = filtered_row

#     return query_neighbor_ids, filtered_neighbor_ids


# def query_embedding_block(config, db_dataset, index,
#                           embeddings, chunk_id_range,
#                           sample_map, n_chunks_per_sample):

#     query_neighbor_ids = []
#     filtered_neighbor_ids = []

#     # Query in sub-blocks.
#     partial_block_size = 1000
#     for partial_start_idx in tqdm(
#             range(0, len(embeddings), partial_block_size),
#             "search",
#     ):
#         partial_end_idx = min(len(embeddings),
#                               partial_start_idx + partial_block_size)
#         partial_embeddings = embeddings[partial_start_idx:partial_end_idx]
#         partial_chunk_id_range = (
#             chunk_id_range[0] + partial_start_idx,
#             chunk_id_range[0] + partial_end_idx,
#         )
#         partial_query_neighbor_ids, partial_filtered_neighbor_ids = \
#             query_embeddings(config, db_dataset, index,
#                              partial_embeddings, partial_chunk_id_range,
#                              sample_map, n_chunks_per_sample,
#                              verbose=False)
#         query_neighbor_ids.append(partial_query_neighbor_ids)
#         filtered_neighbor_ids.append(partial_filtered_neighbor_ids)

#     # Concatenate.
#     query_neighbor_ids = np.concatenate(query_neighbor_ids, axis=0)
#     filtered_neighbor_ids = np.concatenate(filtered_neighbor_ids, axis=0)

#     return query_neighbor_ids, filtered_neighbor_ids


# def query_block_neighbors(config, db_dataset, query_dataset,
#                           index, block):
#     '''Query neighbors of a dataset block (i.e., range).'''

#     n_chunks_per_sample = query_dataset.n_chunks_per_sample

#     # Sample map.
#     sample_ids = sorted(list(set(chunk_id // n_chunks_per_sample
#                                  for chunk_id in range(*block["range"]))))
#     sample_map = {}
#     for i in sample_ids:
#         sample = query_dataset.sample_dataset[i]
#         sample_map[i] = {
#             "dataset_idx" : sample["dataset_id"],
#             "doc_ids" : sample["document_ids"],
#         }

#     # Embed block.
#     embeddings = embed_block(config, query_dataset, block)

#     # Query embeddings.
#     _, filtered_neighbor_ids = query_embedding_block(
#         config, db_dataset, index,
#         embeddings, block["range"],
#         sample_map, n_chunks_per_sample,
#     )

#     # Save neighbors.
#     print_rank_0("save neighbors.")
#     os.makedirs(os.path.dirname(block["path"]), exist_ok=True)
#     f = h5py.File(block["path"], "w")
#     f.create_dataset("neighbors", data=filtered_neighbor_ids)
#     f.close()


# def validate_dataset_neighbors(config, db_dataset,
#                             validate_dataset, num_active_chunks,
#                             prefix, neighbor_dir, index):
#     '''Validate neighbors of each chunk within a dataset.'''

#     def validate(f):
#         assert f["neighbors"].shape[1] == config.retro_validate_num_neighbors_save, \
#             "neighbors.shape == %s; num_neighbors_target == %d." % (
#                 str(f["neighbors"].shape),
#                 config.retro_num_neighbors_target,
#             )
#     blocks = get_blocks_by_rank(
#         neighbor_dir,
#         num_active_chunks,
#         config.retro_block_size,
#         validate=validate,
#     )

#     # Validate each block.
#     for block_index, block in enumerate(blocks.missing):

#         if block is not None:

#             # Progress.
#             print_rank_0("validate '%s' block %d / %d ... %s ... mem %.3f gb, %.1f%%." % (
#                 prefix,
#                 block_index,
#                 len(blocks.missing),
#                 os.path.basename(block["path"]),
#                 psutil.virtual_memory()[3] / 1024**3,
#                 psutil.virtual_memory()[2],
#             ))

#             # Validate block neighbors.
#             validate_block_neighbors(config, db_dataset, validate_dataset, index, block)

#         # Synchronize progress across all ranks. (for easier observation)
#         print_rank_0(" > waiting for other ranks to finish block.")
#         torch.distributed.barrier()


def validate_neighbors(config):
    '''Validate queried neighbors (train & valid).'''

    # Num threads.
    faiss.omp_set_num_threads(64)

    # Load chunk db dataset.
    print_rank_0("load chunk db dataset.")
    db_dataset = get_db_merged_train_dataset(
        project_dir=config.retro_project_dir,
        chunk_length=config.retro_gpt_chunk_length,
        eod_token_id=config.retro_tokenizers.gpt.eod,
    )
    db_dataset.load_doc_tuples()

    # Load index.
    print_rank_0(" > get index.")
    index = get_index(config)

    # Load datasets.
    print_rank_0(" > get dataset map.")
    query_dataset_map = get_query_dataset_map(
        project_dir=config.retro_project_dir,
        gpt_datasets=config.retro_gpt_datasets,
        sample_length=config.retro_gpt_seq_length,
        chunk_length=config.retro_gpt_chunk_length,
    )

    # Query each (i.e., train, valid, test) dataset.
    print_rank_0(" > query.")
    for prefix, info in query_dataset_map.items():
        print_rank_0(" > query '%s' dataset ... %d samples." %
                     (prefix, info["num_active_chunks"]))
        query_dataset_neighbors(config, db_dataset,
                                info["dataset"], info["num_active_chunks"],
                                prefix, info["neighbor_dir"], index)