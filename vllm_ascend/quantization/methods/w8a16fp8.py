#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""W8A16FP8 quantization: FP8 (float8_e4m3fn) weights with BF16 activations.

Weights are stored as FP8 with per-block (128×128) float32 scale_inv.
Dequantization to BF16 is performed at load time, after which inference
uses standard BF16 matrix operations.
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.model_executor.parameter import BlockQuantScaleParameter

from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme
from .registry import register_scheme

# Block size used for per-block quantization scale
_BLOCK_SIZE = 128
_BLOCK_SHAPE = (_BLOCK_SIZE, _BLOCK_SIZE)


def _dequantize_fp8_block(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Dequantize FP8 weight using per-block scale_inv.

    Args:
        weight: FP8 weight tensor of shape [N, K].
        scale_inv: Float32 scale_inv of shape [ceil(N/128), ceil(K/128)].

    Returns:
        BF16 dequantized weight of shape [N, K].
    """
    N, K = weight.shape
    # Expand scale_inv from [N//128, K//128] to [N, K] via repeat_interleave.
    scale_expanded = (
        scale_inv.repeat_interleave(_BLOCK_SIZE, dim=0)[:N, :]
        .repeat_interleave(_BLOCK_SIZE, dim=1)[:, :K]
    )
    return weight.to(torch.bfloat16) * scale_expanded.to(torch.bfloat16)


@register_scheme("W8A16FP8", "linear")
class AscendW8A16FP8LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A16FP8.

    FP8 (float8_e4m3fn) quantized weights with per-block (128×128)
    float32 inverse-scale. Dequantized to BF16 at load time.
    """

    # block_size is used by AscendLinearMethod to set layer.weight_block_size
    # and to create weight_scale_inv as BlockQuantScaleParameter.
    block_size: tuple[int, int] = _BLOCK_SHAPE
    block_scale_params: list[str] = ["weight_scale_inv"]

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
        weight_loader: Callable | None = None,
    ) -> dict[str, Any]:
        n_blocks_out = (output_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        n_blocks_in = (input_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        data = torch.empty(n_blocks_out, n_blocks_in, dtype=torch.float32)
        # Use BlockQuantScaleParameter so MergedColumnParallelLinear weight_loader
        # correctly applies adjust_block_scale_shard when loading.
        if weight_loader is not None:
            param = BlockQuantScaleParameter(
                data=data, input_dim=1, output_dim=0, weight_loader=weight_loader
            )
        else:
            param = data
        return {"weight_scale_inv": param}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequantize FP8 weights to BF16 and discard the scale."""
        weight_bf16 = _dequantize_fp8_block(layer.weight.data, layer.weight_scale_inv.data)
        layer.weight = torch.nn.Parameter(weight_bf16.contiguous(), requires_grad=False)
        layer.weight_scale_inv = None


@register_scheme("W8A16FP8", "moe")
class AscendW8A16FP8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A16FP8.

    FP8 expert weights with per-block float32 inverse-scale.
    Dequantized to BF16 at load time; uses standard BF16 fused_experts
    inference path.
    """

    # Names of block-scale parameters (tells AscendFusedMoEMethod to use GROUP loader)
    block_scale_params: list[str] = ["weight_scale_inv"]

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes,
                dtype=torch.float8_e4m3fn,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        n_blocks_inter = (intermediate_size_per_partition + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        n_blocks_hidden = (hidden_sizes + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        return {
            # scale for w13 (gate+up fused): [E, 2*inter/128, hidden/128]
            "w13_weight_scale_inv": torch.empty(
                num_experts,
                2 * n_blocks_inter,
                n_blocks_hidden,
                dtype=torch.float32,
            ),
            # scale for w2 (down): [E, hidden/128, inter/128]
            "w2_weight_scale_inv": torch.empty(
                num_experts,
                n_blocks_hidden,
                n_blocks_inter,
                dtype=torch.float32,
            ),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ops.fused_moe.experts_selector import (
            select_experts,
            zero_experts_compute,
        )

        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts,
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0),
                global_num_experts - global_redundant_expert_num,
                device=topk_ids.device,
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            log2phy=log2phy,
            mc2_mask=kwargs.get("mc2_mask"),
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dequantize FP8 MoE weights to BF16 and apply unquantized processing."""
        E = layer.w13_weight.shape[0]

        # Dequantize w13 (gate+up) per expert using block-wise scales.
        w13_list = []
        for e in range(E):
            w = _dequantize_fp8_block(layer.w13_weight.data[e], layer.w13_weight_scale_inv.data[e])
            w13_list.append(w)
        w13_bf16 = torch.stack(w13_list, dim=0)  # [E, 2*inter, hidden]

        # Dequantize w2 (down) per expert using block-wise scales.
        w2_list = []
        for e in range(E):
            w = _dequantize_fp8_block(layer.w2_weight.data[e], layer.w2_weight_scale_inv.data[e])
            w2_list.append(w)
        w2_bf16 = torch.stack(w2_list, dim=0)  # [E, hidden, inter]

        # Apply the same transpose + NZ format conversion as unquantized method.
        w13_bf16 = w13_bf16.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_bf16, requires_grad=False)

        w2_bf16 = w2_bf16.transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_bf16, requires_grad=False)

        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

        # Release scale tensors to free memory.
        layer.w13_weight_scale_inv = None
        layer.w2_weight_scale_inv = None
