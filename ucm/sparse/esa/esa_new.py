# TODO: handle preemption
# TODO: init ESA before warmup to make profile_run right!!!
# TODO: reduce memory usage
# TODO: interface of esa_retrieval

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.cuda.nvtx as nvtx

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from ucm.sparse.base import (
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
    UcmSparseBlockManager,
    UcmSparseCpuGpuBuffer
)
from ucm.sparse.utils import get_kv_cache, get_layer_id

import ucm.sparse.esa.esa_interface as esa_lib
esa_retrieval = esa_lib.esa_retrieval
esa_repre = esa_lib.esa_repre
esa_copy = esa_lib.esa_copy
esa_scatter_copy = esa_lib.esa_scatter_copy


@dataclass
class EsaPrefillMetadata:
    """Metadata of current prefill batch."""
    kv_blocks: torch.Tensor  # kv block ids
    repre_blocks: torch.Tensor  # repre block ids
    num_blocks: int = 0  # num of kv blocks


@dataclass
class EsaDecodeMetadata:
    """Metadata of current decode batch."""
    kv_blocks: torch.Tensor  # kv block ids
    repre_blocks: UcmSparseCpuGpuBuffer  # repre block ids
    req_indexes: torch.Tensor  # req indexes
    batch_offset: torch.Tensor  # num repre blocks offset
    num_blocks: int = 0  # num of repre blocks


@dataclass
class EsaMetadata(UcmSparseMetadata):
    """Metadata of current batch."""
    num_prefills: int = 0  # num of prefill reqs
    num_decodes: int = 0  # num of decode reqs
    prefill: Optional[EsaPrefillMetadata] = None
    decode: EsaDecodeMetadata = None


class EsaBlockManager(UcmSparseBlockManager):
    """Manages the allocation of ESA device_repre/host_kv blocks."""

    def __init__(self, capability: int) -> None:
        super().__init__(capability)
        self.req_id_to_blocks: Dict[str, List[int]] = {}

    def allocate(self, num_blocks: int, req_id: str) -> List[int]:
        blocks = super().allocate(num_blocks)
        self.req_id_to_blocks[req_id] = blocks
        return blocks

    def free(self, blocks: List[int], req_id: str) -> None:
        assert req_id in self.req_id_to_blocks
        super().free(blocks)
        del self.req_id_to_blocks[req_id]

    def get_blocks(self, req_id: str) -> List[int]:
        return self.req_id_to_blocks.get(req_id, None)


class ESA(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole) -> None:
        super().__init__(vllm_config, role)

        model_config = vllm_config.model_config
        self.num_kv_heads =  model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device
        self.dtype = model_config.dtype
        self.pin_memory = True

        max_block_per_seq = math.ceil(model_config.max_model_len / vllm_config.cache_config.block_size)
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_blocks =  max_block_per_seq * self.max_num_seqs

        ########################
        # kv and repre cache
        host_kv_shape = (1000, self.block_size, self.num_kv_heads, self.head_size) # TODO:从config里拿到实际的blocks数量*3
        self.host_kv_cache = [
            torch.zeros(host_kv_shape, dtype=self.dtype, device="cpu", pin_memory=self.pin_memory)
            for _ in range(self.num_layers)
        ]
        repre_shape = (self.max_num_blocks, self.num_kv_heads, self.head_size)
        self.device_repre_cache = [
            torch.zeros(repre_shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.block_manager = EsaBlockManager(self.max_num_blocks)
        ########################

        ########################
        # retrieval input and output
        self.retrieval_input = esa_lib.RetrievalInputTensor()
        self.retrieval_output = esa_lib.RetrievalOutputTensor()
        self.retrieval_output.score = torch.zeros(self.max_num_blocks, dtype=self.dtype, device=self.device)
        self.retrieval_output.score_cpu = torch.zeros(self.max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=self.pin_memory)
        self.retrieval_output.score_sorted_cpu = torch.zeros(self.max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=self.pin_memory)
        self.retrieval_output.index_sorted_cpu = torch.zeros(self.max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)
        ########################

        ########################
        # batch dynamic metadata
        self.prefill_kv_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.prefill_repre_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_kv_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_repre_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_req_indexes = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_batch_offset = torch.zeros(self.max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)
        ########################

    def _make_buffer(self,
                     *size: Union[int, torch.SymInt],
                     dtype: torch.dtype,
                     numpy: bool = True) -> UcmSparseCpuGpuBuffer:
        return UcmSparseCpuGpuBuffer(*size,
                                     dtype=dtype,
                                     device=self.device,
                                     pin_memory=self.pin_memory,
                                     with_numpy=numpy)

    def build_sparse_meta(self,
                          scheduler_output: SchedulerOutput,
                          requests: dict[str, CachedRequestState],
                          input_batch: InputBatch,
                          attn_metadata: Any) -> UcmSparseMetadata:
        with nvtx.range(f"esa_build_sparse_meta"):
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))

            num_prefill_kv_blocks = 0
            num_decode_kv_blocks = 0
            num_decode_sparse_blocks = 0
            num_prefills = 0
            num_decodes = 0
            for (req_id, num_scheduled_tokens) in scheduler_output.num_scheduled_tokens.items():
                req = requests[req_id]
                is_decode = num_scheduled_tokens <= 1
                num_blocks = math.ceil(req.num_prompt_tokens / self.block_size)
                block_tables = req.block_ids[0]

                # construct metadata for prefill batch
                is_last_chunk = (not is_decode) and (req.num_computed_tokens + num_scheduled_tokens >= req.num_prompt_tokens)
                if is_last_chunk:
                    assert num_blocks == len(block_tables)
                    # sparse_blocks will be used as indexes for both device_repre_cache and host_kv_cache
                    sparse_blocks = self.block_manager.allocate(num_blocks, req_id=req_id)
                    self.prefill_kv_blocks.np[num_prefill_kv_blocks:num_prefill_kv_blocks + num_blocks] = block_tables
                    self.prefill_repre_blocks.np[num_prefill_kv_blocks:num_prefill_kv_blocks + num_blocks] = sparse_blocks
                    num_prefill_kv_blocks += num_blocks
                    num_prefills += 1

                # construct metadata for decode batch
                if is_decode:
                    sparse_blocks = self.block_manager.get_blocks(req_id)
                    num_sparse_blocks = len(sparse_blocks)
                    assert sparse_blocks is not None, f"req {req_id} does not has sparse blocks"

                    self.decode_kv_blocks.np[num_decode_kv_blocks:num_decode_kv_blocks + num_blocks] = block_tables[:num_blocks]
                    self.decode_repre_blocks.np[num_decode_sparse_blocks:num_decode_sparse_blocks + num_sparse_blocks] = sparse_blocks
                    self.decode_req_indexes.np[num_decodes] = input_batch.req_id_to_index[req_id]
                    num_decode_kv_blocks += num_blocks
                    num_decode_sparse_blocks += num_sparse_blocks
                    num_decodes += 1
                    self.decode_batch_offset[num_decodes] = num_decode_sparse_blocks

            prefill_metadata = None
            decode_metadata = None
            if num_prefills > 0:
                self.prefill_kv_blocks.copy_to_gpu(num_prefill_kv_blocks)
                self.prefill_repre_blocks.copy_to_gpu(num_prefill_kv_blocks)
                prefill_metadata = EsaPrefillMetadata(
                    kv_blocks=self.prefill_kv_blocks.gpu,
                    repre_blocks=self.prefill_repre_blocks.gpu,
                    num_blocks=num_prefill_kv_blocks,
                )

            if num_decodes > 0:
                self.decode_kv_blocks.copy_to_gpu(num_decode_kv_blocks)
                self.decode_repre_blocks.copy_to_gpu(num_decode_sparse_blocks)
                self.decode_req_indexes.copy_to_gpu(num_decodes)
                decode_metadata = EsaDecodeMetadata(
                    kv_blocks=self.decode_kv_blocks.gpu,
                    repre_blocks=self.decode_repre_blocks,
                    req_indexes=self.decode_req_indexes.gpu,
                    batch_offset=self.decode_batch_offset,
                    num_blocks=num_decode_sparse_blocks,
                )

            self.sparse_metadata = EsaMetadata(
                prefill=prefill_metadata,
                decode=decode_metadata,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
            )
            return self.sparse_metadata

    def attention_begin(self,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        layer_name: str,
                        forward_context: ForwardContext,
                        phase: Optional[str] = None) -> None:
        if self.sparse_metadata.decode is None:
            return

        with nvtx.range(f"esa_attention_begin"):
            layer_id = get_layer_id(layer_name)
            self.retrieval_input.query = query
            self.retrieval_input.repre_cache = self.device_repre_cache[layer_id]
            self.retrieval_input.q_index = self.sparse_metadata.decode.req_indexes
            self.retrieval_input.repre_index = self.sparse_metadata.decode.repre_blocks.gpu
            self.retrieval_input.repre_index_cpu = self.sparse_metadata.decode.repre_blocks.cpu
            self.retrieval_input.batch_offset = self.sparse_metadata.decode.batch_offset
            self.retrieval_input.batch = self.sparse_metadata.num_decodes
            self.retrieval_input.s = self.sparse_metadata.decode.num_blocks
            _ = esa_retrieval(self.retrieval_input, self.retrieval_output)

    def attention_finished(self,
                           query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           attn_output: torch.Tensor,
                           layer_name: str,
                           forward_context: ForwardContext,
                           phase: Optional[str] = None) -> None:
        if self.sparse_metadata.prefill is None:
            return

        with nvtx.range(f"esa_attention_finished"):
            layer_id = get_layer_id(layer_name)
            k_cache, _ = get_kv_cache(forward_context, layer_name)
            esa_repre(k_cache.flatten(-2, -1),
                      self.device_repre_cache[layer_id].flatten(-2, -1),
                      self.sparse_metadata.prefill.kv_blocks[:self.sparse_metadata.prefill.num_blocks],
                      self.sparse_metadata.prefill.repre_blocks[:self.sparse_metadata.prefill.num_blocks])
            esa_scatter_copy(k_cache.flatten(-3),
                             self.host_kv_cache[layer_id].flatten(-3),
                             self.sparse_metadata.prefill.kv_blocks[:self.sparse_metadata.prefill.num_blocks],
                             self.sparse_metadata.prefill.repre_blocks[:self.sparse_metadata.prefill.num_blocks])
