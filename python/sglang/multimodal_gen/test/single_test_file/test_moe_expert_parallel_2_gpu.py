"""Expert parallelism must shard experts across ranks and re-form the dense MoE.

Single-process tests can check the arithmetic of the expert split, but not the
two things that only exist with real ranks: that each rank loads a *different*
slice of the checkpoint, and that summing the per-rank partial outputs over the
real collective reproduces the dense block.

Also pins the router-replication guard. EP sums each rank's partial output,
which only equals the dense sum when every rank routes a token to the same
experts. If the routers ever diverge, the ranks compute disjoint halves of two
different routings and the result is plausible but wrong -- silently. The guard
must turn that into an error.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_moe_expert_parallel_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2
_NUM_EXPERTS = 32
_HIDDEN = 64
_INTERMEDIATE = 32
_TOP_K = 8


def _build_block(num_experts: int):
    from sglang.multimodal_gen.runtime.layers.moe import LingBotVideoSparseMoeBlock

    return LingBotVideoSparseMoeBlock(
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_experts=num_experts,
        top_k=_TOP_K,
        score_func="sigmoid",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=None,
    ).cuda()


def _set_router(block, num_experts: int, seed: int) -> None:
    """Router weights come from the checkpoint and are identical on every rank.

    They are allocated with torch.empty, so a test that skips this would leave
    each rank routing on uninitialized memory -- which is exactly the failure
    the replication guard exists to catch.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        block.router.weight.copy_(
            torch.randn(num_experts, _HIDDEN, generator=gen).cuda()
        )
        block.router.e_score_correction_bias.copy_(torch.zeros(num_experts).cuda())


def _worker() -> int:
    from types import SimpleNamespace

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.moe import LingBotVideoGroupedExperts
    from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
        pack_expert_weights,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    # fused_experts reads srt's runtime context and TP group; the GPU worker
    # seeds the first and _sync_srt_tp_group() hands over the second.
    if get_context()._server_args is None:
        get_context().set_server_args(SrtServerArgs(model_path="dummy"))
    set_global_server_args(SimpleNamespace(ep_size=world))
    maybe_init_distributed_environment_and_model_parallel(tp_size=world, sp_size=1)
    ps._sync_srt_tp_group()

    failures = []
    block = _build_block(_NUM_EXPERTS)
    _set_router(block, _NUM_EXPERTS, seed=7)
    info = block.ep_info
    local = _NUM_EXPERTS // world

    if (info.ep_size, info.num_local_experts, info.local_expert_start) != (
        world,
        local,
        rank * local,
    ):
        failures.append(
            f"ep_info wrong: size={info.ep_size} local={info.num_local_experts} "
            f"start={info.local_expert_start}"
        )

    # 1. every rank slices the same checkpoint into its own expert range
    gen = torch.Generator(device="cpu").manual_seed(11)
    full_w1 = torch.randn(_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, generator=gen)
    full_w2 = torch.randn(_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE, generator=gen)
    full_w3 = torch.randn(_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, generator=gen)
    sharded = dict(
        pack_expert_weights(
            iter(
                [
                    ("blocks.0.ffn.experts.w1", full_w1),
                    ("blocks.0.ffn.experts.w2", full_w2),
                    ("blocks.0.ffn.experts.w3", full_w3),
                ]
            ),
            ep_info=info,
        )
    )
    w13_local = sharded["blocks.0.ffn.experts.w13_weight"]
    start = info.local_expert_start
    expected = torch.cat((full_w1, full_w3), dim=1)[start : start + local]
    if not torch.equal(w13_local, expected):
        failures.append(f"rank loaded the wrong expert range [{start}:{start + local}]")

    # ranks must hold *different* data -- a partition bug that gave every rank
    # experts [0:local] would still halve memory and still change the output
    checksum = torch.tensor(
        [w13_local.double().sum()], device="cuda", dtype=torch.float64
    )
    gathered = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(gathered, checksum)
    values = [float(g.item()) for g in gathered]
    if len(set(values)) != world:
        failures.append(f"ranks hold identical expert shards: {values}")

    # 2. sharded forward + all-reduce == dense forward over all experts
    with torch.no_grad():
        block.experts.w13_weight.copy_(w13_local.cuda())
        block.experts.w2.copy_(sharded["blocks.0.ffn.experts.w2"].cuda())

    x = torch.randn(
        1, 128, _HIDDEN, device="cuda", dtype=torch.bfloat16, generator=None
    )
    dist.broadcast(x, src=0)  # identical activations, as TP replication gives
    with torch.no_grad():
        ep_out = block(x)

    dense = _build_block(_NUM_EXPERTS)
    dense.router = block.router
    dense.ep_enabled = False
    dense.num_local_experts = _NUM_EXPERTS
    dense._router_replication_checked = True
    dense.experts = LingBotVideoGroupedExperts(
        _NUM_EXPERTS, _HIDDEN, _INTERMEDIATE
    ).cuda()
    with torch.no_grad():
        dense.experts.w13_weight.copy_(torch.cat((full_w1, full_w3), dim=1).cuda())
        dense.experts.w2.copy_(full_w2.cuda())
        dense_out = dense(x)

    delta = (ep_out.float() - dense_out.float()).abs().max().item()
    scale = dense_out.float().abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        ep_out.float().flatten(), dense_out.float().flatten(), dim=0
    ).item()
    # One bf16 ulp: EP rounds each rank's partial sum, dense accumulates the
    # top-k slots in fp32 and rounds once. See the Expert Parallelism doc.
    if cosine < 0.9999 or delta / scale > 1e-2:
        failures.append(f"EP != dense: cosine={cosine:.8f} rel={delta / scale:.3e}")

    # 3. the router-replication guard must catch divergent routers
    diverged = _build_block(_NUM_EXPERTS)
    _set_router(diverged, _NUM_EXPERTS, seed=7 + rank)  # differs per rank
    try:
        with torch.no_grad():
            diverged(x)
    except RuntimeError as exc:
        if "router" not in str(exc):
            failures.append(f"guard raised the wrong error: {exc}")
    else:
        failures.append("router replication guard did not fire on divergent routers")

    for failure in failures:
        print(f"FAILURE rank{rank}: {failure}", flush=True)
    return 1 if failures else 0


class TestMoeExpertParallelTwoGpu(CustomTestCase):
    def test_expert_parallel_two_ranks(self):
        if not current_platform.is_cuda():
            self.skipTest("srt's Triton MoE path is CUDA-only")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        procs = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29753",
                }
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, __file__],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [p.communicate(timeout=600)[0] for p in procs]
        codes = [p.returncode for p in procs]
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()
