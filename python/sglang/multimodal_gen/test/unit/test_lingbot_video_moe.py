# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEArchConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.layers.moe import (
    NON_LOCAL_EXPERT_ID,
    LingBotVideoGroupedExperts,
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
    MoeExpertParallelInfo,
    to_local_expert_ids,
)
from sglang.multimodal_gen.runtime.models.dits import (
    lingbot_video_moe as dits_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
    LingBotVideoAttention,
    LingBotVideoTransformer3DModel,
    _joint_position_ids,
    is_expert_parallel_param,
    make_joint_position_ids,
    pack_expert_weights,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

_LINGBOT_MODULE_SUBDIRS = (
    "scheduler",
    "text_encoder",
    "processor",
    "transformer",
    "vae",
)


def test_moe_path_resolves_moe_configs():
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "lingbot-video-moe-30b-a3b")
        os.makedirs(model_dir)
        with open(
            os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"_class_name": "LingBotVideoPipeline", "_diffusers_version": "0.37.1"},
                f,
            )
        for subdir in _LINGBOT_MODULE_SUBDIRS:
            os.mkdir(os.path.join(model_dir, subdir))
        info = get_model_info(model_dir, backend="sglang")

    assert info.pipeline_cls.__name__ == "LingBotVideoPipeline"
    assert info.pipeline_config_cls is LingBotVideoMoEPipelineConfig
    assert info.sampling_param_cls is LingBotVideoMoESamplingParams


def test_arch_config_defaults_without_mlp_only_layers():
    arch = LingBotVideoMoEArchConfig()
    assert arch.num_experts == 128
    assert arch.mlp_only_layers == ()


def test_router_bias_shifts_selection_but_not_gate_weights():
    router = LingBotVideoRouter(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=False,
        n_group=None,
        topk_group=None,
        route_scale=1.0,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, 0.0],
                    [-4.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

    top_indices, top_scores = router(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    assert set(top_indices[0].tolist()) == {0, 3}
    raw = torch.sigmoid(torch.tensor([4.0, -4.0]))
    picked = {
        int(idx): float(score.detach())
        for idx, score in zip(top_indices[0], top_scores[0])
    }
    assert abs(picked[0] - float(raw[0])) < 1e-5
    assert abs(picked[3] - float(raw[1])) < 1e-5


def _sdpa(q, k, v, attn_mask=None, attn_mask_meta=None):
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_mask
    )
    return out.transpose(1, 2)


def _real_attention(num_heads, head_dim):
    attn = object.__new__(LingBotVideoAttention)
    attn.local_num_heads = num_heads
    attn.head_dim = head_dim
    attn.to_q = attn.to_k = attn.to_v = attn.to_out = lambda x: (x, None)
    attn.norm_q = attn.norm_k = lambda t: t
    attn.attn = _sdpa
    return attn


def test_attention_isolates_samples_across_batch(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 3, 8
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)

    valid = [seq_len, seq_len - 2, seq_len - 5]
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i, length in enumerate(valid):
        mask[i, :length] = True

    batched = attn.forward(x, (freqs, freqs), mask)

    for i, length in enumerate(valid):
        solo = attn.forward(
            x[i : i + 1],
            (freqs[i * seq_len : (i + 1) * seq_len],) * 2,
            mask[i : i + 1],
        )
        torch.testing.assert_close(batched[i : i + 1, :length], solo[:, :length])

    # Flattening the batch into one sequence lets sample 0 attend across the
    # boundary; its output must differ from the isolated per-sample result.
    flat = attn.forward(x.reshape(1, batch * seq_len, hidden), (freqs, freqs), None)
    flat = flat.reshape(batch, seq_len, hidden)
    assert (flat[0, : valid[0]] - batched[0, : valid[0]]).abs().max() > 1e-3


def test_attention_forwards_2d_mask_and_varlen_metadata(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 2, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    captured = {}

    def capture_attention(q, k, v, attn_mask=None, attn_mask_meta=None):
        captured["mask"] = attn_mask
        captured["meta"] = attn_mask_meta
        return _sdpa(q, k, v, attn_mask=attn_mask)

    attn.attn = capture_attention
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    metadata = {"max_seqlen": seq_len}

    attn.forward(x, (freqs, freqs), mask, metadata)

    assert captured["mask"] is mask
    assert captured["meta"] is metadata


def test_attention_single_sample_matches_direct_attention(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, seq_len = 4, 8, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, hidden)
    freqs = torch.zeros(seq_len, head_dim // 2)

    out = attn.forward(x, (freqs, freqs), attention_mask=None)

    qkv = x.unflatten(2, (num_heads, head_dim))
    expected = _sdpa(qkv, qkv, qkv).flatten(2)
    torch.testing.assert_close(out, expected)


class _FakeBatchEncoding(dict):
    def to(self, _device):
        return self


class _FakeQwenProcessor:
    def __init__(self, prompt_width, prefix_width, true_len):
        self.prompt_width = prompt_width
        self.prefix_width = prefix_width
        self.true_len = true_len

    def __call__(self, **kwargs):
        if "max_length" in kwargs:
            width = self.prompt_width
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, : self.true_len] = 1
        else:
            width = self.prefix_width
            mask = torch.ones(1, width, dtype=torch.long)
        return _FakeBatchEncoding(
            input_ids=torch.zeros(1, width, dtype=torch.long),
            attention_mask=mask,
        )


def _text_encoding_stage(processor, encoder):
    stage = object.__new__(LingBotVideoTextEncodingStage)
    stage.text_encoders = [encoder]
    stage.tokenizers = [processor]
    stage.token_length = 128
    stage.hidden_state_skip_layer = 0
    stage.prompt_template = PROMPT_TEMPLATE
    stage._crop_start = None
    return stage


def test_text_encoding_crops_template_then_trims_padding():
    prompt_width, prefix_width, true_len, channels = 10, 3, 8, 4
    hidden = torch.arange(prompt_width, dtype=torch.float32)
    hidden = hidden.view(1, prompt_width, 1).expand(1, prompt_width, channels)

    def encoder(**kwargs):
        return SimpleNamespace(hidden_states=[hidden])

    stage = _text_encoding_stage(
        _FakeQwenProcessor(prompt_width, prefix_width, true_len), encoder
    )
    embeds, mask = stage._encode_prompt(
        "a structured caption", torch.device("cpu"), torch.float32
    )

    assert tuple(embeds.shape) == (1, true_len - prefix_width, channels)
    torch.testing.assert_close(embeds, hidden[:, prefix_width:true_len])
    assert int(mask.sum()) == true_len - prefix_width
    assert stage._compute_crop_start() == prefix_width


def test_check_inputs_enforces_frame_and_size_contract():
    check = LingBotVideoTextEncodingStage.check_inputs
    check(480, 832, 1)
    check(480, 832, 81)
    try:
        check(480, 832, 82)
        raise AssertionError("expected ValueError for num_frames=82")
    except ValueError:
        pass
    try:
        check(480, 830, 81)
        raise AssertionError("expected ValueError for width=830")
    except ValueError:
        pass


def test_decode_scale_and_shift_invert_vae_normalization():
    config = LingBotVideoMoEPipelineConfig()
    scale, shift = config.get_decode_scale_and_shift(
        torch.device("cpu"), torch.float32, vae=None
    )
    arch = config.vae_config.arch_config
    std = torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    mean = torch.tensor(arch.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(scale, 1.0 / std)
    torch.testing.assert_close(shift, mean)


def test_latents_stay_fp32_under_bf16_precision():
    config = LingBotVideoMoEPipelineConfig()
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32


def test_grouped_experts_store_packed_w13_weight():
    experts = LingBotVideoGroupedExperts(
        num_experts=2, hidden_size=4, intermediate_size=3
    )
    names = {n for n, _ in experts.named_parameters()}
    assert "w13_weight" in names and "w2" in names
    assert "w1" not in names and "w3" not in names
    assert tuple(experts.w13_weight.shape) == (2, 6, 4)  # [E, 2I, H]


def test_preprocess_packs_w1_w3_into_w13_weight():
    pack = _packer(ep_size=1)
    E, I, H = 2, 3, 4
    w1 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H)
    w2 = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I)
    w3 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H) + 100.0
    # block 0: w1 before w3; block 1: w3 before w1 (order-independence).
    src = [
        ("blocks.0.ffn.experts.w1", w1),
        ("blocks.0.ffn.experts.w2", w2),
        ("blocks.0.ffn.experts.w3", w3),
        ("blocks.0.ffn.router.weight", torch.zeros(E, H)),
        ("blocks.1.ffn.experts.w3", w3.clone()),
        ("blocks.1.ffn.experts.w2", w2.clone()),
        ("blocks.1.ffn.experts.w1", w1.clone()),
    ]
    out = dict(pack(iter(src)))
    assert set(out.keys()) == {
        "blocks.0.ffn.experts.w13_weight",
        "blocks.0.ffn.experts.w2",
        "blocks.0.ffn.router.weight",
        "blocks.1.ffn.experts.w13_weight",
        "blocks.1.ffn.experts.w2",
    }
    packed = torch.cat((w1, w3), dim=1)  # gate then up, dim-1
    torch.testing.assert_close(out["blocks.0.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.1.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.0.ffn.experts.w2"], w2)


# ---------------------------------------------------------------------------
# Expert parallelism
# ---------------------------------------------------------------------------


# The unit suite also runs on the AMD/ROCm lane, where torch.cuda.is_available()
# is True -- so availability alone would not skip there. These cases drive srt's
# CUDA Triton MoE kernels and its NCCL symmetric-memory allocation, which is not
# what that lane is meant to cover.
requires_cuda_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or current_platform.is_hip(),
    reason="srt's Triton MoE + symmetric-memory path is CUDA-only",
)


def _ep_info(ep_size: int, ep_rank: int, num_experts: int) -> MoeExpertParallelInfo:
    num_local = num_experts // ep_size
    return MoeExpertParallelInfo(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_experts=num_local,
        local_expert_start=ep_rank * num_local,
    )


def _packer(*, ep_size: int, ep_rank: int = 0, num_experts: int = 2):
    ep_info = _ep_info(ep_size, ep_rank, num_experts)
    return lambda it: pack_expert_weights(it, ep_info=ep_info)


def test_grouped_experts_allocate_only_local_experts():
    experts = LingBotVideoGroupedExperts(
        num_experts=8, hidden_size=4, intermediate_size=3, num_local_experts=2
    )
    assert experts.num_experts == 8
    assert tuple(experts.w13_weight.shape) == (2, 6, 4)  # [E_local, 2I, H]
    assert tuple(experts.w2.shape) == (2, 4, 3)  # [E_local, H, I]


def test_local_expert_ids_drop_non_local_to_minus_one():
    num_experts, ep_size, ep_rank = 8, 4, 2  # this rank owns experts 4 and 5
    ep_info = _ep_info(ep_size, ep_rank, num_experts)
    top_indices = torch.arange(num_experts, dtype=torch.int32).reshape(2, 4)

    local_ids = to_local_expert_ids(top_indices, ep_info)

    expected = torch.tensor(
        [
            [
                NON_LOCAL_EXPERT_ID,
                NON_LOCAL_EXPERT_ID,
                NON_LOCAL_EXPERT_ID,
                NON_LOCAL_EXPERT_ID,
            ],
            [0, 1, NON_LOCAL_EXPERT_ID, NON_LOCAL_EXPERT_ID],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(local_ids, expected)
    # The sentinel must be exactly -1: moe_align_block_size shifts ids by +1 into
    # an (E_local + 1)-entry histogram, so any other out-of-range id is an
    # out-of-bounds access rather than a filtered row.
    assert int(local_ids.min()) == NON_LOCAL_EXPERT_ID
    assert int(local_ids.max()) < ep_info.num_local_experts


def test_expert_parallel_shards_tile_the_dense_checkpoint():
    E, I, H, ep_size = 8, 3, 4, 4
    w1 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H)
    w2 = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I)
    w3 = w1 + 100.0
    router = torch.randn(E, H)
    src = [
        ("blocks.0.ffn.experts.w1", w1),
        ("blocks.0.ffn.experts.w2", w2),
        ("blocks.0.ffn.experts.w3", w3),
        ("blocks.0.ffn.router.weight", router),
    ]

    shards = [
        dict(_packer(ep_size=ep_size, ep_rank=r, num_experts=E)(iter(src)))
        for r in range(ep_size)
    ]

    for shard in shards:
        assert shard["blocks.0.ffn.experts.w13_weight"].shape[0] == E // ep_size
        assert shard["blocks.0.ffn.experts.w2"].shape[0] == E // ep_size
        # The router stays full and replicated.
        torch.testing.assert_close(shard["blocks.0.ffn.router.weight"], router)

    for key, dense in (
        ("blocks.0.ffn.experts.w13_weight", torch.cat((w1, w3), dim=1)),
        ("blocks.0.ffn.experts.w2", w2),
    ):
        torch.testing.assert_close(torch.cat([s[key] for s in shards], dim=0), dense)


def test_is_expert_parallel_param_matches_only_expert_weights():
    assert is_expert_parallel_param("blocks.0.ffn.experts.w13_weight", None)
    assert is_expert_parallel_param("blocks.7.ffn.experts.w2", None)
    assert not is_expert_parallel_param("blocks.0.ffn.router.weight", None)
    assert not is_expert_parallel_param(
        "blocks.0.ffn.shared_experts.up_proj.weight", None
    )
    assert not is_expert_parallel_param("blocks.0.attn.to_q.weight", None)


def _make_moe_block(**overrides):
    kwargs = dict(
        hidden_size=8,
        intermediate_size=4,
        num_experts=8,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
    )
    kwargs.update(overrides)
    return LingBotVideoSparseMoeBlock(**kwargs)


def test_moe_block_without_expert_parallel_owns_every_expert():
    # Also the only coverage of resolve_moe_expert_parallel's disabled path,
    # which must not touch the (uninitialized) EP group.
    block = _make_moe_block()
    assert not block.ep_enabled
    assert block.ep_info.ep_size == 1
    assert block.num_local_experts == 8
    assert block.experts.w13_weight.shape[0] == 8


@pytest.mark.parametrize("ep_rank", [0, 3])
def test_moe_block_with_expert_parallel_owns_only_its_shard(monkeypatch, ep_rank):
    import sglang.multimodal_gen.runtime.layers.moe as moe_module

    monkeypatch.setattr(
        moe_module, "get_global_server_args", lambda: SimpleNamespace(ep_size=4)
    )
    monkeypatch.setattr(moe_module, "get_ep_world_size", lambda: 4)
    monkeypatch.setattr(moe_module, "get_ep_rank", lambda: ep_rank)

    block = _make_moe_block()

    assert block.ep_enabled
    assert block.num_local_experts == 2
    assert block.ep_info.local_expert_start == ep_rank * 2
    assert block.experts.w13_weight.shape[0] == 2
    assert block.experts.w2.shape[0] == 2
    # The router stays whole: every rank must reach the same routing decision.
    assert block.router.weight.shape[0] == 8


def test_expert_parallel_rejects_indivisible_expert_count(monkeypatch):
    import sglang.multimodal_gen.runtime.layers.moe as moe_module

    monkeypatch.setattr(
        moe_module, "get_global_server_args", lambda: SimpleNamespace(ep_size=3)
    )
    monkeypatch.setattr(moe_module, "get_ep_world_size", lambda: 3)
    monkeypatch.setattr(moe_module, "get_ep_rank", lambda: 0)

    with pytest.raises(ValueError, match="divisible"):
        _make_moe_block(num_experts=8)


@pytest.fixture(scope="module")
def srt_single_rank_process_group():
    """Seed srt's runtime context and TP group for a single-process fused_experts.

    fused_experts allocates its output under `use_symmetric_memory(get_tp_group())`
    (srt/.../fused_moe.py), so it needs srt's `_TP` even at world size 1. In the
    real runtime the diffusion `initialize_model_parallel` hands its own group over
    via `_sync_srt_tp_group()` (runtime/distributed/parallel_state.py); only this
    single-process test has to build one itself.
    """
    import torch.distributed as dist

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import destroy_model_parallel
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if get_context()._server_args is None:
        get_context().set_server_args(SrtServerArgs(model_path="dummy"))

    already_initialized = dist.is_initialized()
    if not already_initialized:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    if not already_initialized:
        destroy_model_parallel()
        dist.destroy_process_group()


def _dense_vs_ep_partial_sums(
    *, num_tokens, E, I, H, top_k, ep_size, seed, partial_output_dtype=None
):
    """Run fused_experts dense, then once per EP rank, and sum the partials.

    Returns (ep_total, dense). With ``partial_output_dtype=None`` the partials are
    bf16 and summed in bf16 -- the Phase-1 path, accurate to ~1 bf16 ulp. With
    ``partial_output_dtype=torch.float32`` each rank emits an fp32 partial, the
    partials are summed in fp32 and rounded once -- the production EP path, which
    is bit-exact to the dense baseline (see `LingBotVideoSparseMoeBlock.forward`).
    """
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    torch.manual_seed(seed)
    device = torch.device("cuda")
    tokens = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16)
    w13 = torch.randn(E, 2 * I, H, device=device, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(E, H, I, device=device, dtype=torch.bfloat16) * 0.1
    top_indices = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    top_scores = torch.rand(num_tokens, top_k, device=device)

    def run(w13_local, w2_local, topk_ids, num_local_experts, part_dtype=None):
        runner_config = MoeRunnerConfig(
            num_experts=E,
            num_local_experts=num_local_experts,
            hidden_size=H,
            intermediate_size_per_partition=I,
            top_k=top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )
        topk_output = StandardTopKOutput(
            topk_weights=top_scores.float(),
            topk_ids=topk_ids,
            router_logits=torch.empty(0, device=device),
        )
        return fused_experts(
            tokens.contiguous(),
            w13_local,
            w2_local,
            topk_output,
            runner_config,
            partial_output_dtype=part_dtype,
        )

    dense = run(w13, w2, top_indices, E)  # reference: default bf16 output

    num_local = E // ep_size
    acc_dtype = partial_output_dtype or dense.dtype
    ep_total = torch.zeros(dense.shape, dtype=acc_dtype, device=dense.device)
    for rank in range(ep_size):
        ep_info = _ep_info(ep_size, rank, E)
        start = ep_info.local_expert_start
        ep_total += run(
            w13[start : start + num_local].contiguous(),
            w2[start : start + num_local].contiguous(),
            to_local_expert_ids(top_indices, ep_info).to(torch.int32),
            num_local,
            part_dtype=partial_output_dtype,
        )
    if partial_output_dtype is not None:
        ep_total = ep_total.to(dense.dtype)  # single round after the fp32 sum
    return ep_total, dense


@requires_cuda_moe
@pytest.mark.parametrize("ep_size", [4, 8])
def test_expert_parallel_partial_sums_are_exact_at_top_k_2(
    srt_single_rank_process_group, ep_size
):
    """Phase-1 correctness: dropping non-local experts to -1 and summing the
    per-rank partials reproduces the dense MoE.

    This case is bit-exact, but only because top_k == 2: srt combines the two
    slots with a plain `torch.add` of already-bf16 rows (fused_moe.py, the
    `topk == 2 and routed_scaling_factor == 1.0` branch), which is exactly what
    the per-rank split reproduces. Do NOT read this as a general guarantee --
    see the top_k=8 test below for what the production config actually gets.
    """
    ep_total, dense = _dense_vs_ep_partial_sums(
        num_tokens=64, E=8, I=32, H=16, top_k=2, ep_size=ep_size, seed=0
    )
    torch.testing.assert_close(ep_total, dense, rtol=0, atol=0)


@requires_cuda_moe
def test_expert_parallel_partial_sums_match_dense_at_production_top_k(
    srt_single_rank_process_group,
):
    """The same claim at the shipped config (128 experts, top_k 8), where the
    result is accurate to a bf16 ulp rather than bit-exact.

    At top_k > 2 srt accumulates the slots in fp32 inside `moe_sum_reduce` and
    rounds once, while EP rounds each rank's partial sum before they are added.
    Measured on H100: max|delta| one bf16 ulp, ~6e-3 relative, cosine 0.9999933 --
    consistent across ep_size 2/4/8. EP_PLAN budgets ~1e-2 relative on bf16.
    """
    ep_total, dense = _dense_vs_ep_partial_sums(
        num_tokens=1024, E=128, I=512, H=1024, top_k=8, ep_size=8, seed=1
    )
    max_delta = (ep_total.float() - dense.float()).abs().max().item()
    assert max_delta / dense.float().abs().max().item() < 1e-2
    cosine = torch.nn.functional.cosine_similarity(
        ep_total.float().flatten(), dense.float().flatten(), dim=0
    ).item()
    assert cosine > 0.99999


@requires_cuda_moe
@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_expert_parallel_fp32_partial_is_bit_exact_at_production_top_k(
    srt_single_rank_process_group, ep_size
):
    """The production EP path: fp32 partial output + fp32 cross-rank sum + a single
    round is BIT-EXACT to the dense baseline, at top_k=8 and every ep_size.

    This is the fix for the bf16 double-rounding measured in the test above: the
    per-rank partial is no longer rounded to bf16 before the sum, so EP reproduces
    the dense single-rounding exactly and the result is independent of ep_size.
    Requires `fused_experts`' `partial_output_dtype` mode.
    """
    ep_total, dense = _dense_vs_ep_partial_sums(
        num_tokens=1024, E=128, I=512, H=1024, top_k=8, ep_size=ep_size, seed=1,
        partial_output_dtype=torch.float32,
    )
    torch.testing.assert_close(ep_total, dense, rtol=0, atol=0)


@requires_cuda_moe
def test_expert_parallel_disabled_is_bit_identical(srt_single_rank_process_group):
    """ep_size == 1 must leave the dense path untouched (no filtering at all)."""
    ep_total, dense = _dense_vs_ep_partial_sums(
        num_tokens=256, E=128, I=512, H=1024, top_k=8, ep_size=1, seed=2
    )
    torch.testing.assert_close(ep_total, dense, rtol=0, atol=0)


def test_joint_position_ids_match_reference_and_cover_padding():
    dev = torch.device("cpu")
    gt, gh, gw = 2, 3, 4
    n_video = gt * gh * gw

    # B==1, no padding: byte-identical to the per-sample reference.
    vec = _joint_position_ids(torch.tensor([5]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(vec, make_joint_position_ids(5, gt, gh, gw, dev))

    # B==1 with padding: real tokens match the text_len=4 reference; the extra
    # padding row is (0,0,0). vec has n_video+L rows (matches q for B*S).
    vec_p = _joint_position_ids(torch.tensor([4]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(
        vec_p[: n_video + 4], make_joint_position_ids(4, gt, gh, gw, dev)
    )
    torch.testing.assert_close(
        vec_p[n_video + 4 :], torch.zeros((1, 3), dtype=torch.int32)
    )

    # B>1 with padding: covers B*S rows; each sample's real tokens match its ref.
    text_lens = [5, 3, 6]
    B, L = len(text_lens), 6
    vec_b = _joint_position_ids(torch.tensor(text_lens), gt, gh, gw, L, dev)
    assert vec_b.shape[0] == B * (n_video + L)
    for i, t in enumerate(text_lens):
        start = i * (n_video + L)
        real = n_video + t
        torch.testing.assert_close(
            vec_b[start : start + real], make_joint_position_ids(t, gt, gh, gw, dev)
        )
