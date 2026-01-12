# TODO: handle preemption
# TODO: init ESA before warmup to make profile_run right!!!
# TODO: reduce memory usage
# TODO: interface of esa_retrieval

import math
from itertools import chain
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.cuda.nvtx as nvtx

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from ucm.utils import Config
from ucm.sparse.base import (
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseCachedRequestData,
    UcmSparseRole,
    UcmSparseBlockManager,
    UcmSparseCpuGpuBuffer
)
from ucm.sparse.utils import cdiv, get_kv_cache, get_layer_id

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


@dataclass
class EsaDecodeMetadata:
    """Metadata of current decode batch."""
    kv_blocks: torch.Tensor  # kv block ids
    repre_blocks: UcmSparseCpuGpuBuffer  # repre block ids
    leftover_kv_blocks: torch.Tensor  # leftover kv block ids
    leftover_repre_blocks: torch.Tensor  # leftover repre block ids
    req_indexes: torch.Tensor  # req indexes
    batch_offset: torch.Tensor  # num repre blocks offset
    fixed_block_indexes: torch.Tensor  # fixed block indexes in repre_blocks (with maximum score)
    num_blocks: int = 0  # num of repre blocks
    num_leftover_blocks: int = 0  # num of leftover blocks
    num_fixed_blocks: int = 0  # num of fixed blocks (with maximum score)


@dataclass
class EsaMetadata(UcmSparseMetadata):
    """Metadata of current batch."""
    num_prefills: int = 0  # num of prefill reqs
    num_decodes: int = 0  # num of decode reqs
    prefill: Optional[EsaPrefillMetadata] = None
    decode: Optional[EsaDecodeMetadata] = None


@dataclass
class EsaCachedRequestData(UcmSparseCachedRequestData):
    """Metadata of cached request."""
    sparse_blocks: List[int] = field(default_factory=list)  # block ids of device_repre/host_kv 
    num_prompt_blocks: int = 0  # num of prompt blocks when prefill
    num_compressed_prompt_blocks: int = 0  # num of prompt blocks when decode (after compression)
    step: int = 0  # the step of request


class ESA(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole) -> None:
        super().__init__(vllm_config, role)

        model_config = vllm_config.model_config
        max_block_per_seq = math.ceil(model_config.max_model_len / vllm_config.cache_config.block_size)
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_blocks =  max_block_per_seq * self.max_num_seqs
        self.num_kv_heads =  model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device
        self.dtype = model_config.dtype
        self.pin_memory = True

        self._init_sparse_cfg()
        self._init_cache()

    def _init_sparse_cfg(self):
        self.esa_cfg = Config(self.vllm_config.kv_transfer_config).get_config().get("ucm_sparse_config").get("ESA")
        self.min_blocks = self.esa_cfg.get("min_blocks", 4)
        self.sparse_ratio = self.esa_cfg.get("sparse_ratio", 0.2)
        self.init_window = self.esa_cfg.get("init_window_sz", 1)
        self.local_window = self.esa_cfg.get("local_window_sz", 2)
        self.fixed_window = self.init_window + self.local_window
        assert self.min_blocks > self.fixed_window, "ESA min_blocks should be larger than init_window_sz + local_window_sz."

    def _init_cache(self):
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
        self.block_manager = UcmSparseBlockManager(self.max_num_blocks)

        # retrieval input and output
        self.retrieval_input = esa_lib.RetrievalInputTensor()
        self.retrieval_output = esa_lib.RetrievalOutputTensor()
        self.retrieval_output.score = torch.zeros(self.max_num_blocks, dtype=self.dtype, device=self.device)
        self.retrieval_output.score_cpu = torch.zeros(self.max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=self.pin_memory)
        self.retrieval_output.score_sorted_cpu = torch.zeros(self.max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=self.pin_memory)
        self.retrieval_output.index_sorted_cpu = torch.zeros(self.max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)

        # batch dynamic metadata
        self.prefill_kv_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.prefill_repre_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_kv_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_repre_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_leftover_kv_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_leftover_repre_blocks = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_req_indexes = self._make_buffer(self.max_num_blocks, dtype=torch.int32)
        self.decode_fixed_indexes = self._make_buffer(self.fixed_window * self.max_num_seqs, dtype=torch.int32)
        self.decode_batch_offset = torch.zeros(self.max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)

    def _make_buffer(self,
                     *size: Union[int, torch.SymInt],
                     dtype: torch.dtype,
                     numpy: bool = True) -> UcmSparseCpuGpuBuffer:
        return UcmSparseCpuGpuBuffer(*size,
                                     dtype=dtype,
                                     device=self.device,
                                     pin_memory=self.pin_memory,
                                     with_numpy=numpy)

    def _clear_buffer(self) -> None:
        self.prefill_kv_blocks.clear()
        self.prefill_repre_blocks.clear()
        self.decode_kv_blocks.clear()
        self.decode_repre_blocks.clear()
        self.decode_leftover_kv_blocks.clear()
        self.decode_leftover_repre_blocks.clear()
        self.decode_req_indexes.clear()
        self.decode_fixed_indexes.clear()

    def _get_num_compressed_prompt_blocks(self, num_prompt_blocks: int) -> int:
        return self.fixed_window + int((num_prompt_blocks - self.fixed_window) * self.sparse_ratio)

    def build_sparse_meta(self,
                          scheduler_output: SchedulerOutput,
                          requests: dict[str, CachedRequestState],
                          input_batch: InputBatch,
                          attn_metadata: Any) -> UcmSparseMetadata:
        with nvtx.range(f"esa_build_sparse_meta"):
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))

            num_prefills = 0
            num_decodes = 0
            num_decode_sparse_blocks = 0
            self._clear_buffer()
            for (req_id, num_scheduled_tokens) in scheduler_output.num_scheduled_tokens.items():
                req = requests[req_id]
                is_decode = req_id in self.cached_reqs
                if is_decode:
                    num_prompt_blocks = self.cached_reqs[req_id].num_prompt_blocks
                else:
                    num_prompt_blocks = cdiv(req.num_prompt_tokens, self.block_size)
                    if num_prompt_blocks <= self.min_blocks:
                        continue
                block_tables = req.block_ids[0]
                num_blocks = len(block_tables)

                # construct metadata for prefill batch
                is_last_chunk = (not is_decode) and (req.num_computed_tokens + num_scheduled_tokens >= req.num_prompt_tokens)
                if is_last_chunk:
                    # sparse_blocks will be used as indexes for both device_repre_cache and host_kv_cache
                    sparse_blocks = self.block_manager.allocate(num_blocks)
                    if sparse_blocks is None:
                        continue

                    self.cached_reqs[req_id] = EsaCachedRequestData(sparse_blocks=sparse_blocks,
                                                                    num_prompt_blocks=num_prompt_blocks)
                    self.prefill_kv_blocks.append_numpy(block_tables)
                    self.prefill_repre_blocks.append_numpy(sparse_blocks)
                    num_prefills += 1

                # construct metadata for decode batch
                if is_decode:
                    if req_id not in self.cached_reqs:
                        continue

                    sparse_blocks = self.cached_reqs[req_id].sparse_blocks
                    num_sparse_blocks = len(sparse_blocks)
                    assert num_sparse_blocks > 0, f"req {req_id} does not has sparse blocks."

                    is_first_decode = self.cached_reqs[req_id].step == 0
                    if is_first_decode:
                        num_compressed_prompt_blocks = self._get_num_compressed_prompt_blocks(num_prompt_blocks)
                        self.cached_reqs[req_id].num_compressed_prompt_blocks = num_compressed_prompt_blocks
                        num_leftover_blocks = num_blocks - num_compressed_prompt_blocks + 1
                        self.decode_leftover_kv_blocks.append_numpy(block_tables[-num_leftover_blocks:])
                        self.decode_leftover_repre_blocks.append_numpy(sparse_blocks[-num_leftover_blocks:])
                    else:
                        num_compressed_prompt_blocks = self.cached_reqs[req_id].num_compressed_prompt_blocks
                    self.cached_reqs[req_id].step += 1

                    fixed_indexes = list(chain(range(num_decode_sparse_blocks + self.init_window),
                                               range(num_decode_sparse_blocks + num_sparse_blocks - self.local_window,
                                                     num_decode_sparse_blocks + num_sparse_blocks - 1)))
                    self.decode_kv_blocks.append_numpy(block_tables[:num_compressed_prompt_blocks - 1])
                    self.decode_repre_blocks.append_numpy(sparse_blocks[:num_prompt_blocks - 1])
                    self.decode_fixed_indexes.append_numpy(fixed_indexes)
                    self.decode_req_indexes.append_numpy([input_batch.req_id_to_index[req_id]])
                    num_decode_sparse_blocks += num_prompt_blocks - 1
                    num_decodes += 1
                    self.decode_batch_offset[num_decodes] = num_decode_sparse_blocks

            prefill_metadata = None
            decode_metadata = None
            if num_prefills > 0:
                self.prefill_kv_blocks.copy_to_gpu()
                self.prefill_repre_blocks.copy_to_gpu()
                prefill_metadata = EsaPrefillMetadata(
                    kv_blocks=self.prefill_kv_blocks.valid_gpu,
                    repre_blocks=self.prefill_repre_blocks.valid_gpu,
                )

            if num_decodes > 0:
                self.decode_kv_blocks.copy_to_gpu()
                self.decode_repre_blocks.copy_to_gpu()
                self.decode_req_indexes.copy_to_gpu()
                self.decode_fixed_indexes.copy_to_gpu()
                decode_metadata = EsaDecodeMetadata(
                    kv_blocks=self.decode_kv_blocks.gpu,
                    repre_blocks=self.decode_repre_blocks,
                    leftover_kv_blocks=self.decode_kv_blocks.gpu,
                    leftover_repre_blocks=self.decode_repre_blocks.gpu,
                    req_indexes=self.decode_req_indexes.gpu,
                    batch_offset=self.decode_batch_offset,
                    fixed_block_indexes=self.decode_fixed_indexes.gpu,
                    num_blocks=num_decode_sparse_blocks,
                    num_leftover_blocks=self.decode_leftover_kv_blocks.size,
                    num_fixed_blocks=self.decode_fixed_indexes.size,
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
                      self.sparse_metadata.prefill.kv_blocks,
                      self.sparse_metadata.prefill.repre_blocks)
            esa_scatter_copy(k_cache.flatten(-3),
                             self.host_kv_cache[layer_id].flatten(-3),
                             self.sparse_metadata.prefill.kv_blocks,
                             self.sparse_metadata.prefill.repre_blocks)
