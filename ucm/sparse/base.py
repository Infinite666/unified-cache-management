"""
UcmSparseBase Class provides interfaces for general sparse attention algorithm implementation in vLLM.

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to retrieval/load KV cache.
        estimate_num_slots_sparsed() - get the number of required slots.
        update_state_after_alloc() - update UcmSparse state after
            temporary buffer alloc by the CacheManager.
        request_finished_in_scheduler() - called when a request is finished, with
            the computed kv cache blocks for the request.
            Returns metadata for the next step.

    Worker-side: runs in each worker, retrieval/load KV cache.
        execute_begin() - hook at the beginning of "ModelRunner->execute_model->model".
        execute_finished() - hook at the end of "ModelRunner->execute_model->model".
        attention_begin() - hook at the beginning of "unified_attention".
        attention_finished() - hook at the end of "unified_attention".
        update_states() - hook at the beginning of "ModelRunner->execute_model".
"""

from __future__ import annotations

import enum
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


import torch
import numpy as np

from ucm.logger import init_logger


logger = init_logger(__name__)

INVALID_SLOT = -1


class UcmSparseRole(enum.Enum):
    # sparser running in the scheduler process
    SCHEDULER = 0

    # sparser running in the worker process
    WORKER = 1


class UcmSparseMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to hold the necessary metadata for sparse.
    """
    pass


class UcmSparseCachedRequestData(ABC):
    """
    Abstract Metadata used to cache necessary data of requests.
    """
    pass


class UcmSparseBlockManager:
    """Manages the allocation of blocks."""

    def __init__(self, capability: int) -> None:
        self.capability = capability
        self.free_blocks = list(range(capability - 1, -1, -1))

    def allocate(self, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            logger.warning(f"No enough free sparse blocks for {num_blocks}")
            return None

        res = []
        for _ in range(num_blocks):
            res.append(self.free_blocks.pop())
        return res

    def free(self, blocks: List[int]) -> None:
        for block in blocks:
            self.free_blocks.append(block)


class UcmSparseCpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU. Inferred by vLLM."""

    def __init__(
        self,
        *size: Union[int, torch.SymInt],
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = True,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size,
                               dtype=dtype,
                               device="cpu",
                               pin_memory=pin_memory)
        self.gpu = self.cpu.to(device)
        self.np: np.ndarray
        self.n = 0

        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call UcmSparseCpuGpuBuffer with with_numpy=False")
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: Optional[int] = None) -> None:
        # TODO: replace with esa_copy
        if n is None:
            n = self.n
        if n <= 0:
            return
        self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: Optional[int] = None) -> None:
        # TODO: replace with esa_copy
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            n = self.n
        if n <= 0:
            return
        self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)

    def append_numpy(self, data: List[Any]) -> None:
        size = len(data)
        assert self.np is not None, "append_numpy meed to be initialized by with_numpy=True."
        assert self.n + size < self.cpu.shape[0], "append_numpy data out of range."
        self.np[self.n:self.n + size] = data
        self.n += size

    def clear(self) -> None:
        self.n = 0

    @property
    def size(self) -> int:
        return self.n

    @property
    def valid_np(self) -> np.ndarray:
        return self.np[:self.n]

    @property
    def valid_cpu(self) -> torch.Tensor:
        return self.cpu[:self.n]

    @property
    def valid_gpu(self) -> torch.Tensor:
        return self.gpu[:self.n]

class UcmSparseBase(ABC):
    """
    An general interface for impl sparse attention algorithm in vLLM
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole) -> None:
        self.cached_reqs: dict[str, UcmSparseCachedRequestData] = dict()
        self.sparse_metadata: Optional[UcmSparseMetadata] = None
        self.vllm_config = vllm_config
        self.role = role

    # ==============================
    # Worker-side methods
    # ==============================

    def initialize_sparse_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        This is called at the "Worker->initialize_from_config" function.
        Initializes the necessary caches for sparse.
        """
        pass

    def get_sparse_cache_size(self) -> Tuple[int, int]:
        """
        This is called at the "Worker->determine_available_memory" function.
        Returns the fixed_cache_size and compression_ratio (relative compression ratio of one block).
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput) -> None:
        """
        This is called at the beginning of "ModelRunner->execute_model->model" function.
        """
        pass

    def execute_finished(self) -> None:
        """
        This is called at the end of "ModelRunner->execute_model" function.
        """
        pass

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        """
        This is called at the beginning of "unified_attention".
        Sparse attention algorithm can modify forward_context.attn_metadata if necessary.
        (UC_TODO: modify dataclass is not allowed in python?)
        """
        pass

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        """
        This is called at the end of "unified_attention".
        """
        pass

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        This is called at the beginning of "ModelRunner->execute_model" function.
        Update the cached states with the scheduler output.
        """
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]) -> None:
        """
        This is called at the beginning of "Scheduler->add_request" function.
        """
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]) -> None:
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        """
        This is called by "Scheduler->schedule" function to estimate the number of required blocks.
        """
        return INVALID_SLOT

    def update_state_after_alloc(self, request: Request, num_blocks: int) -> None:
        """
        Update UcmSparse state after block allocation.
        """
        pass

    def build_sparse_meta(self,
                          scheduler_output: SchedulerOutput,
                          requests: dict[str, CachedRequestState],
                          input_batch: InputBatch,
                          attn_metadata: Any) -> UcmSparseMetadata:
        """
        Build the sparse metadata for this step.
        """
        pass

    def allocate_slots(self,
                       kv_cache_manager: KVCacheManager,
                       request: Request,
                       num_slots_sparsed: int) -> None:
        pass
