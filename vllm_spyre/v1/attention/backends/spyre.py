""" Attention layer with simple KV caching without paging.
Uses SDPA for attention
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)

from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm.logger import init_logger

logger = init_logger(__name__)

class SpyreSDPABackend(AttentionBackend):
    accept_output_buffer: bool = False
    @staticmethod
    def get_name() -> str:
        return "Spyre_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["SpyreSDPABackendImpl"]:
        return SpyreSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["SpyreSDPAMetadata"]:
        return SpyreSDPAMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> Type["SpyreSDPAMetadataBuilder"]:
        return SpyreSDPAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
    ) -> Tuple[int, ...]:
        # No paging, taking max seq len of 128
        max_seq_len = 128
        max_batch_size = 1
        return (2, max_batch_size, num_kv_heads, max_seq_len, head_size)

@dataclass
class SpyreSDPAMetadata():
    num_tokens = 0
    bsize = 1

    # For prefill
    is_prefill: bool

    # For decode
    past_token = 0

    def __init__(self, num_tokens, bsize, is_prefill, masks, past_token):
        self.num_tokens = num_tokens
        self.bsize = bsize
        self.is_prefill = is_prefill
        self.past_token = past_token
        self.masks = masks

class SpyreSDPAMetadataBuilder(AttentionMetadataBuilder[SpyreSDPAMetadata]):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(self, data) -> SpyreSDPAMetadata:
        num_tokens = len(data.input_tokens)
        is_prefill = data.is_prompt
        bsize = len(data.input_positions)
        past_token = 0 #[0 * bsize]
        # TODO: Handle batch sizes later
        if not is_prefill:
            past_token = data.input_masks.shape[2] - 1 # bsize x qsize x kvsize

        masks = data.input_masks

        attn_metadata = SpyreSDPAMetadata(
            num_tokens=num_tokens,
            bsize=bsize,
            is_prefill=is_prefill,
            past_token=past_token,
            masks=masks
        )

        return attn_metadata

class SpyreSDPABackendImpl(AttentionImpl[SpyreSDPAMetadata]):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]] = None,
            logits_soft_cap: Optional[float] = None,
            attn_type: str = AttentionType.DECODER,
            use_irope: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Check for supported head sizes
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        self.attn_type = attn_type
        # if attn_type != AttentionType.DECODER:
        #     print("Encoder self-attention and "
        #                               "encoder/decoder cross-attention "
        #                               "are not implemented for "
        #                               "SpyreSDPABackendImpl")

        self.sdpa_compute_prefill = self._sdpa_compute_op
        self.sdpa_compute_decode = self._sdpa_compute_op

    def _sdpa_store_op(
            self,
            keys: torch.Tensor,
            values: torch.Tensor,
            key_cache: Optional[torch.Tensor],
            value_cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        if key_cache is not None and value_cache is not None and value_cache.numel() > 0:
            #print(f"{keys.shape}, {key_cache.shape}")
            key_cache_result = torch.cat((key_cache, keys), dim=2).to(dtype=keys.dtype)
            value_cache_result = torch.cat((value_cache, values), dim=2).to(dtype=keys.dtype)
            return (
                key_cache_result,
                value_cache_result,
                key_cache_result,
                value_cache_result,
            )
        else:
            return (keys, values, keys, values)

    def _sdpa_compute_op(
            self,
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            nheads: int,
            kvheads: int,
            p_dropout: float,
            scale_factor: Optional[float],
            attn_metadata: SpyreSDPAMetadata,
            ) -> torch.Tensor:
        queries = query.transpose(2, 1)

        if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
            key_cache = key_cache.transpose(2, 1)
            value_cache = value_cache.transpose(2, 1)

        mask = attn_metadata.masks
        if mask is not None:
            while len(mask.size()) != 4:
                mask = mask.unsqueeze(1)

        # Expand kv so black-box attn will work
        expansion = nheads // kvheads
        # k/v: b h l d
        if expansion != 1:
            keys_e = key_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = value_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        else:
            keys_e = key_cache
            values_e = value_cache

        attn_mask = mask
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(dtype=queries.dtype)

        is_causal = attn_metadata.is_prefill and (mask is None and not (key_cache.shape[2] != 1 and queries.shape[2] == 1))

        # print(queries.shape)
        # print(keys_e.shape)
        attn = F.scaled_dot_product_attention(
            queries,
            keys_e,
            values_e,
            attn_mask=attn_mask,
            dropout_p=p_dropout,
            is_causal=is_causal,
            scale=scale_factor,
        )

        # DEBUG 
        # attn_weight = queries @ keys_e.transpose(-2, -1) * 0.1
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        # attn = attn_weight @ values_e
        # attn = queries

        # attn: bs x seq_len x nheads*emb_v_per_head
        # attn: b x h x qlen x ds
        # attn after permute: b x qlen x h x ds
        # b x qlen x (d)
        attn = attn.transpose(2, 1).contiguous()
        return attn

    def forward(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: SpyreSDPAMetadata,  # type: ignore
            output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [batch_size, num_tokens, num_heads * head_size]
            key: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            value: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        bsize = attn_metadata.bsize
        # Reshape the query, key, and value tensors.
        queries = query.view(bsize, -1, self.num_heads, self.head_size)
        q_len = query.shape[1]

        keys = key.view(bsize, -1, self.num_kv_heads, self.head_size).to(dtype=queries.dtype)
        values = value.view(bsize, -1, self.num_kv_heads, self.head_size).to(dtype=queries.dtype)

        past_token = attn_metadata.past_token
        if attn_metadata.is_prefill:
            past_key_value_state = (None, None)
        else:
            past_key_value_state = kv_cache[:,:,:,:past_token,:]

        keys_compute, values_compute, keys_return, values_return = (
            self._sdpa_store_op(
                keys,
                values,
                past_key_value_state[0],
                past_key_value_state[1],
            )
        )
        #keys_compute, values_compute = keys, values # DEBUG

        if attn_metadata.is_prefill:
            attn = self.sdpa_compute_prefill(
                queries,
                keys_compute,
                values_compute,
                self.num_heads,
                self.num_kv_heads,
                0.0, # dropout
                self.scale,
                attn_metadata,
            )
        else:
            attn = self.sdpa_compute_decode(
                queries,
                keys_compute,
                values_compute,
                self.num_heads,
                self.num_kv_heads,
                0.0,
                self.scale,
                attn_metadata,
            )

        attn = attn.view(-1, self.num_heads * self.head_size)

        # Store back KV cache

        kv_len = keys_return.shape[2]
        kv_cache[0,:,:,:kv_len,:] = keys_return
        kv_cache[1,:,:,:kv_len,:] = values_return

        return attn
