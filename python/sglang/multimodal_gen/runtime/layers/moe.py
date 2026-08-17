# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.distributed import (
    get_ep_group,
    get_ep_rank,
    get_ep_world_size,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

logger = logging.getLogger(__name__)

# Expert id handed to srt's fused_experts for a token routed to an expert that
# this rank does not own. moe_align_block_size shifts ids by +1 into an
# (E + 1)-entry histogram, so -1 lands in bucket 0 and the triton GEMMs
# zero-fill the block; any other out-of-range id is an out-of-bounds access.
NON_LOCAL_EXPERT_ID = -1


class MoeExpertParallelInfo(msgspec.Struct, frozen=True):
    """Contiguous expert -> rank partition for MoE expert parallelism."""

    ep_size: int
    ep_rank: int
    num_local_experts: int
    local_expert_start: int

    @property
    def enabled(self) -> bool:
        return self.ep_size > 1


def resolve_moe_expert_parallel(num_experts: int) -> MoeExpertParallelInfo:
    """Resolve the expert shard owned by this rank.

    EP is its own orthogonal parallel axis (`get_ep_group()`): experts are
    sharded across it while attention/norms/router replicate. Tokens reach the
    MoE block replicated across the EP group and the router is bit-identical
    across those ranks, so slicing experts and summing the partial outputs with
    an all-reduce over the EP group reproduces the dense result exactly.
    """
    # The short-circuit is load-bearing: get_ep_world_size() *raises* when the
    # EP group is not initialized -- it does not fall back to 1 -- so reading it
    # unconditionally would break every single-process unit test that builds a
    # MoE block. With EP off there is nothing to resolve anyway.
    ep_size = get_ep_world_size() if get_global_server_args().ep_size > 1 else 1
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )
    num_local_experts = num_experts // ep_size
    ep_rank = get_ep_rank() if ep_size > 1 else 0
    return MoeExpertParallelInfo(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_experts=num_local_experts,
        local_expert_start=ep_rank * num_local_experts,
    )


@functools.lru_cache(maxsize=None)
def assert_ep_safe_triton_moe(
    *, num_local_experts: int, intermediate_size: int, top_k: int
) -> None:
    """Refuse to run EP on srt Triton MoE configurations that mis-handle filtering.

    Slicing the experts flips on srt's `filter_expert`, and two of its code paths
    write the zeros for a filtered block through the wrong indexing:

    - `enable_fused_moe_sum_all_reduce` (topk > 2) points the store at the
      combined [num_tokens, hidden] output while `write_zeros_to_output` still
      indexes rows by token*topk, so it writes past the tensor and clobbers other
      experts' contributions.
    - A tuned `*_down.json` carrying `USE_TMA` makes the up-GEMM output
      expert-sorted, but the zero-store keeps using unsorted row indices.

    Neither fires on the default config, so this is a tripwire rather than a live
    failure -- but both are silent corruption, which is exactly what must not be
    discovered from a slightly-wrong video.

    Checked lazily on the first forward, not in __init__: srt's `exec` config
    namespace is not published yet at model-construction time, and this reads the
    same flag `fused_experts` itself reads per call. lru_cache keeps it to one
    real check per shape.
    """
    from sglang.srt.runtime_context import get_exec

    if top_k > 2 and get_exec().moe.enable_fused_moe_sum_all_reduce:
        raise ValueError(
            "expert parallelism is incompatible with "
            "--enable-fused-moe-sum-all-reduce at top_k > 2: srt's filtered-block "
            "zero-store indexes the combined output by token*topk and writes out "
            "of bounds. Disable that flag or run with --ep-size 1."
        )

    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _moe_support_tma,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_moe_configs,
        )

        if not _moe_support_tma():
            return
        # Mirrors srt's own lookup: E, _, N = w2.shape, i.e. the *local* expert
        # count and the per-expert intermediate size (fused_moe.py:412-421).
        down_configs = get_moe_configs(
            num_local_experts, intermediate_size, None, 0, 0, down_moe=True
        )
    except Exception:  # pragma: no cover - srt internals moved; guard is advisory
        logger.warning(
            "Could not check srt's down-projection MoE config for USE_TMA; "
            "expert parallelism assumes the non-TMA down path."
        )
        return

    if down_configs and any(cfg.get("USE_TMA") for cfg in down_configs.values()):
        raise ValueError(
            "expert parallelism is incompatible with a USE_TMA down-projection "
            "MoE config: srt writes the zeros for a filtered expert block at "
            "unsorted row indices into an expert-sorted buffer. Remove the tuned "
            "config or run with --ep-size 1."
        )


def to_local_expert_ids(
    top_indices: torch.Tensor, ep_info: MoeExpertParallelInfo
) -> torch.Tensor:
    """Map global expert ids to this rank's local ids, dropping the rest.

    Equivalent to srt's `local_expert_mapping` gather table
    (`srt/layers/moe/token_dispatcher/standard.py`) for a contiguous partition,
    but as pure arithmetic — the model is built on the meta device, where a
    registered mapping buffer would need lazy materialization.
    """
    local_ids = top_indices - ep_info.local_expert_start
    return torch.where(
        (local_ids >= 0) & (local_ids < ep_info.num_local_experts),
        local_ids,
        torch.full_like(local_ids, NON_LOCAL_EXPERT_ID),
    )


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        route_scale: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(num_experts), persistent=True
        )

    def _group_limited_topk(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(seq_len, self.n_group, experts_per_group)
            .reshape(seq_len, -1)
        )
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype)


class LingBotVideoGroupedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = (
            num_experts if num_local_experts is None else num_local_experts
        )
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, intermediate_size)
        )


class LingBotVideoSparseMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        n_shared_experts: int | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.ep_info = resolve_moe_expert_parallel(num_experts)
        # Plain scalars for the forward path: this block is torch.compile'd
        # (_compile_conditions in lingbot_video_moe.py), and Dynamo has no
        # handling for a msgspec.Struct property, so reading ep_info directly in
        # forward would graph-break on every call.
        self.ep_enabled = self.ep_info.enabled
        self.num_local_experts = self.ep_info.num_local_experts
        self.router = LingBotVideoRouter(
            hidden_size,
            num_experts,
            top_k,
            score_func,
            norm_topk_prob,
            n_group,
            topk_group,
            routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(
            num_experts,
            hidden_size,
            intermediate_size,
            num_local_experts=self.ep_info.num_local_experts,
        )
        self.shared_experts: LingBotVideoMLP | None = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size, intermediate_size * n_shared_experts
            )
        self._router_replication_checked = not self.ep_enabled

    def _check_router_is_replicated(self) -> None:
        """Fail loudly if the ranks would route the same token differently.

        EP re-forms the dense result by summing each rank's partial output, which
        is only the dense sum if every rank selected the *same* experts for a
        token. That holds because the router is replicated and its input is
        TP-identical -- but nothing else enforces it, and a violation is silent:
        the ranks compute disjoint halves of two different routings, sum them,
        and produce a plausible but wrong result.
        """
        self._router_replication_checked = True
        checksum = torch.stack(
            [
                self.router.weight.double().sum(),
                self.router.e_score_correction_bias.double().sum(),
            ]
        )
        gathered = get_ep_group().all_gather(checksum.reshape(1, -1), dim=0)
        if not torch.equal(gathered, gathered[:1].expand_as(gathered)):
            raise RuntimeError(
                "MoE router weights differ across the expert parallel group "
                f"(per-rank checksums: {gathered.tolist()}). Expert parallelism "
                "sums each rank's partial output, which only reconstructs the "
                "dense result when every rank routes a token to the same experts."
            )

    def _run_sglang_triton_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        partial_output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        # Under EP the non-local experts are dropped here; fused_experts writes
        # hard zeros for them (before mul_routed_weight), so their top_scores
        # contribute nothing and the per-rank partial sums add up to the dense
        # result. num_local_experts != num_experts flips on `filter_expert`.
        if self.ep_enabled:
            assert_ep_safe_triton_moe(
                num_local_experts=self.num_local_experts,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
            )
            top_indices = to_local_expert_ids(top_indices, self.ep_info)
        topk_output = StandardTopKOutput(
            topk_weights=top_scores.float(),
            topk_ids=top_indices.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        # Router pre-scales the topk scores; fused_experts must not apply routed_scaling_factor.
        runner_config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=self.top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )
        out = fused_experts(
            tokens.contiguous().bfloat16(),
            self.experts.w13_weight.bfloat16(),
            self.experts.w2.bfloat16(),
            topk_output,
            runner_config,
            partial_output_dtype=partial_output_dtype,
        )
        # In the EP fp32-partial path, keep the fp32 partial so the caller can sum
        # across ranks and round exactly once (matching the dense single-rounding).
        if partial_output_dtype is not None:
            return out
        return out.type_as(tokens)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        if not self._router_replication_checked:
            # Once per block, on the first forward -- the weights are loaded by
            # then, and every rank reaches this in the same order.
            self._check_router_is_replicated()
        top_indices, top_scores = self.router(tokens)
        if self.ep_enabled:
            # Each rank computes the partial sum over the experts it owns, then the
            # partials are summed across the EP group. Emitting the partial in fp32
            # and rounding once *after* the all-reduce reproduces the dense path's
            # single rounding exactly (bit-exact, and independent of ep_size);
            # rounding each rank's partial to bf16 first would double-round.
            #
            # The all-reduce must precede shared_experts: that output is replicated,
            # so summing it too would scale it by ep_size.
            out = self._run_sglang_triton_experts(
                tokens, top_scores, top_indices, partial_output_dtype=torch.float32
            )
            out = get_ep_group().all_reduce(out)
            out = out.to(tokens.dtype)
        else:
            out = self._run_sglang_triton_experts(tokens, top_scores, top_indices)
        out = out.reshape(b, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
