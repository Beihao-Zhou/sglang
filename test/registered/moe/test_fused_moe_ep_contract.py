"""Regression tests for filtered-expert and FP32 partial-output MoE paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner import triton as triton_runner
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    outplace_fused_experts,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


def _dummy_moe_inputs(device="cpu"):
    hidden_states = torch.empty((2, 8), device=device, dtype=torch.bfloat16)
    w1 = torch.empty((4, 16, 8), device=device, dtype=torch.bfloat16)
    w2 = torch.empty((4, 8, 8), device=device, dtype=torch.bfloat16)
    topk_weights = torch.empty((2, 2), device=device, dtype=torch.float32)
    topk_ids = torch.empty((2, 2), device=device, dtype=torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty(0, device=device),
    )
    return hidden_states, w1, w2, topk_output


class TestFusedMoeExpertParallelContract(CustomTestCase):
    def test_partial_output_fake_dtype_matches_runtime_contract(self):
        with FakeTensorMode():
            hidden_states, w1, w2, topk_output = _dummy_moe_inputs(device="cuda")
            output = outplace_fused_experts(
                hidden_states,
                w1,
                w2,
                topk_output.topk_weights,
                topk_output.topk_ids,
                partial_output_dtype=torch.float32,
            )
            with self.assertRaisesRegex(ValueError, "no_combine=False"):
                outplace_fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    no_combine=True,
                    partial_output_dtype=torch.float32,
                )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, torch.float32)

    def test_partial_output_rejects_unsupported_contracts(self):
        hidden_states, w1, w2, topk_output = _dummy_moe_inputs()
        cases = (
            (
                torch.bfloat16,
                MoeRunnerConfig(inplace=False),
                "only supports torch.float32",
            ),
            (torch.float32, MoeRunnerConfig(inplace=True), "inplace=False"),
            (
                torch.float32,
                MoeRunnerConfig(inplace=False, no_combine=True),
                "no_combine=False",
            ),
            (
                torch.float32,
                MoeRunnerConfig(inplace=False, routed_scaling_factor=1.5),
                "routed_scaling_factor",
            ),
        )

        for partial_output_dtype, config, expected_message in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, expected_message):
                    fused_moe.fused_experts(
                        hidden_states,
                        w1,
                        w2,
                        topk_output,
                        config,
                        partial_output_dtype=partial_output_dtype,
                    )

    def test_filtered_expert_routing_disables_down_tma(self):
        aligned = (
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )

        results = {}
        for filter_expert in (False, True):
            up_config = {"BLOCK_SIZE_M": 16}
            down_config = {"BLOCK_SIZE_M": 16, "USE_TMA": True}
            with (
                patch.object(
                    fused_moe,
                    "try_get_optimal_moe_config",
                    return_value=(up_config, (down_config, None)),
                ),
                patch.object(fused_moe, "_moe_support_tma", return_value=True),
                patch.object(fused_moe, "moe_align_block_size", return_value=aligned),
            ):
                results[filter_expert] = fused_moe._prepare_fused_moe_run(
                    torch.zeros((1, 4), dtype=torch.bfloat16),
                    torch.zeros((2, 8, 4), dtype=torch.bfloat16),
                    torch.zeros((2, 4, 4), dtype=torch.bfloat16),
                    torch.zeros((1, 1), dtype=torch.int32),
                    use_fp8_w8a8=False,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=None,
                    filter_expert=filter_expert,
                )

        self.assertTrue(results[False][2])
        self.assertFalse(results[True][2])

    def test_filtered_and_partial_paths_disable_fused_sum_all_reduce(self):
        common = dict(
            enabled=True,
            no_combine=False,
            topk=4,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        )

        self.assertTrue(
            fused_moe._should_use_fused_moe_sum_all_reduce(
                **common, partial_output_dtype=None, filter_expert=False
            )
        )
        self.assertFalse(
            fused_moe._should_use_fused_moe_sum_all_reduce(
                **common, partial_output_dtype=None, filter_expert=True
            )
        )
        self.assertFalse(
            fused_moe._should_use_fused_moe_sum_all_reduce(
                **common,
                partial_output_dtype=torch.float32,
                filter_expert=False,
            )
        )

    def test_standard_to_triton_pre_permute_passes_filter_state(self):
        hidden_states, w1, w2, topk_output = _dummy_moe_inputs()
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
        quant_info = TritonMoeQuantInfo(w13_weight=w1, w2_weight=w2)
        prepared = (
            {"BLOCK_SIZE_M": 16},
            None,
            False,
            False,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )
        observed = []

        def prepare(*args, **kwargs):
            observed.append(kwargs["filter_expert"])
            return prepared

        with patch.object(fused_moe, "_prepare_fused_moe_run", side_effect=prepare):
            for num_local_experts in (4, 8):
                triton_runner.pre_permute_standard_to_triton(
                    dispatch_output,
                    quant_info,
                    MoeRunnerConfig(num_experts=8, num_local_experts=num_local_experts),
                    running_state={},
                )

        self.assertEqual(observed, [True, False])

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
    def test_fp32_partial_sum_matches_dense_single_rounding(self):
        from sglang.srt.runtime_context import get_context

        server_args_override = get_context().override_server_args()
        server_args_override.install()
        self.addCleanup(server_args_override.restore)
        torch.manual_seed(1)
        device = "cuda"
        num_tokens, num_experts, topk = 32, 8, 4
        hidden_size, intermediate_size, ep_size = 64, 64, 2

        hidden_states = torch.randn(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        w1 = torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
        w2 = torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        router_logits = torch.randn((num_tokens, num_experts), device=device)
        topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), topk)
        topk_ids = topk_ids.to(torch.int32)

        def run(local_w1, local_w2, local_ids, config, partial_output_dtype=None):
            return fused_moe.fused_experts(
                hidden_states,
                local_w1,
                local_w2,
                StandardTopKOutput(topk_weights, local_ids, router_logits),
                config,
                partial_output_dtype=partial_output_dtype,
            )

        single_rank_group = SimpleNamespace(world_size=1)
        with (
            patch.object(fused_moe, "get_tp_group", return_value=single_rank_group),
            patch.object(fused_moe, "is_allocation_symmetric", return_value=False),
        ):
            dense = run(
                w1,
                w2,
                topk_ids,
                MoeRunnerConfig(
                    num_experts=num_experts,
                    num_local_experts=num_experts,
                    inplace=False,
                ),
            )

            num_local_experts = num_experts // ep_size
            partial_sum = torch.zeros_like(dense, dtype=torch.float32)
            for rank in range(ep_size):
                start = rank * num_local_experts
                local_ids = topk_ids - start
                local_ids[(local_ids < 0) | (local_ids >= num_local_experts)] = -1
                partial = run(
                    w1[start : start + num_local_experts].contiguous(),
                    w2[start : start + num_local_experts].contiguous(),
                    local_ids,
                    MoeRunnerConfig(
                        num_experts=num_experts,
                        num_local_experts=num_local_experts,
                        inplace=False,
                    ),
                    partial_output_dtype=torch.float32,
                )
                self.assertEqual(partial.dtype, torch.float32)
                partial_sum.add_(partial)

        torch.testing.assert_close(partial_sum.to(dense.dtype), dense, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
