# EP draft branch — status, bugs fixed, problems met, next steps

Working notes for `beihao/moe-dit-ep` (draft). Scope: Phase 1 of `EP_PLAN.md` —
expert parallelism for the LingBot-Video MoE DiT, EP group == TP group, no token
dispatch.

User-facing documentation of the feature itself lives in
`docs/docs/sglang-diffusion/expert_parallelism.mdx`. This file is the engineering
record: what was broken, what was learned, and what is left.

**State:** implementation complete and validated end-to-end on 2xH100. 41 unit
tests plus a 2-GPU parity test pass (from 16 passed / 1 failed at the start of
review). Not yet reviewed by anyone else.

---

## 1. Bugs fixed

### 1.1 The new GPU parity test failed outright

`test_expert_parallel_partial_sums_match_dense_experts` died with
`AssertionError: tensor model parallel group is not initialized`.

`fused_experts` allocates its output under
`use_symmetric_memory(get_tp_group(), ...)`, so it needs srt's `_TP` **even at
world size 1**. The test seeded srt's server args but not its process group. In
the real runtime this never surfaces because the diffusion
`initialize_model_parallel` hands its own group over via `_sync_srt_tp_group()`
(`runtime/distributed/parallel_state.py`); only a single-process test has to
build one itself.

Fixed with a module-scoped fixture that does
`init_distributed_environment(world_size=1, ...)` + `initialize_model_parallel(1)`
and tears both down.

### 1.2 The parity test asserted a guarantee the model does not have

The test asserted `rtol=0, atol=0` — bit-exactness — and passed. That was
misleading: it used `top_k=2`, where srt combines the two slots with a plain
`torch.add` of already-bf16 rows, a form the per-rank split reproduces exactly.
The shipped model uses `top_k=8`, where srt switches to the fp32-accumulating
`moe_sum_reduce` and EP is **not** bit-exact (~1 bf16 ulp).

As written the test advertised bit-exact EP to anyone who read it. Split into
two tests: the `top_k=2` case keeps `rtol=0, atol=0` with a comment explaining
that the exactness is structural and does not generalize, plus a new
`top_k=8, num_experts=128` case asserting ulp-level tolerance.

### 1.3 The EP construction path had zero test coverage

No test constructed a `LingBotVideoSparseMoeBlock`, so
`resolve_moe_expert_parallel()` — the function that reads the server args and the
TP group and decides this rank's expert range — was never executed by any test.
Added coverage for the disabled path, the sharded path at two ranks, and the
indivisible-expert-count error.

### 1.4 The FSDP double-shard guard could not detect its own failure mode

`_assert_params_not_sharded` checks that no ignored parameter came back as a
`DTensor`. But if the *predicate* matched nothing — a stale name pattern, e.g.
`EXPERT_PARAM_SUFFIXES` drifting from the module layout — then `ignored_params`
was never populated, FSDP sharded the experts anyway, **and the assert still
passed**, because it iterated the same empty match list.

That is precisely the scenario the guard exists for, and it was invisible.
`_resolve_fsdp_ignored_params` now raises when conditions are declared but match
no parameter. Covered by a new test in `test_fsdp_load.py`.

### 1.5 The `ep_size != tp_size` error did not explain the real trap

`--num-gpus N` alone resolves to `tp_size=1` with every remaining GPU assigned to
`sp_degree`, so `--ep-size N` was rejected with a bare constraint restatement.
The error now says why (EP sums partials across ranks holding the *same* tokens;
SP ranks hold different slices, which needs all-to-all dispatch) and what to do
(pass `--tp-size` explicitly). Verified live.

### 1.6 Dynamo graph-break in the compiled block

`LingBotVideoBlock` is `torch.compile`d, and the forward path read
`self.ep_info.enabled` — a `msgspec.Struct` property, which Dynamo has no
handling for. Hoisted `enabled` and `num_local_experts` to plain `bool`/`int`
attributes in `__init__`. This also satisfies the repo's init-static-value
extraction convention.

### 1.7 The new CUDA tests would have run on the AMD CI lane

The `unit` suite is glob-discovered (`test/server/gpu_cases.py`
`_discover_unit_tests()` globs `unit/test_*.py`), so new tests are picked up with
no registration — but that same suite is run by **three** lanes: CUDA
(`1-gpu-h100`), AMD MI300, and ROCm 7.2.0.

Under ROCm `torch.cuda.is_available()` returns **True**, so a plain
`skipif(not torch.cuda.is_available())` would *not* skip there: the new tests
would execute on MI300, driving srt's CUDA Triton MoE kernels and NCCL
symmetric-memory allocation, in a lane whose comments state the unit suite is
"portable, CPU-style… don't require NVIDIA hardware". Best case they pass; worst
case this branch breaks someone else's lane.

Replaced the three CUDA gates with a shared `requires_cuda_moe` marker that also
excludes ROCm via `current_platform.is_hip()`.

### 1.8 Two srt Triton bugs — guarded, not fixed

Found while verifying the sentinel (see 1.10). Both are reachable **only** when EP is on
(`filter_expert=True`) and both are off by default, so they are latent rather
than live — but both are silent corruption, not crashes:

- **`enable_fused_moe_sum_all_reduce` at `top_k > 2`**: the store is pointed at
  the combined `[num_tokens, hidden]` output, while `write_zeros_to_output` still
  indexes rows by `token * topk`. That writes past the tensor and clobbers other
  experts' already-accumulated contributions.
- **A tuned `*_down.json` carrying `USE_TMA`**: the up-GEMM output is
  expert-sorted (`c_sorted=down_moe_use_tma`), but the filtered-block zero-store
  keeps using unsorted row indices, racing with valid rows.

Nothing disables TMA for `filter_expert` — only for LoRA hooks. The MoE block now
refuses both configurations up front with a message naming the flag. **These
remain unfixed upstream; see next steps.**

### 1.9 The 2-GPU test would never have run in CI

Unlike the glob-discovered `unit` suite (1.7), `single_test_file/` tests are
**explicitly listed**: a file must appear in both `STANDALONE_FILES["2-gpu"]` and
`STANDALONE_FILE_EST_TIMES` in `test/server/gpu_cases.py` or it is simply never
executed. Two opposite traps in one directory tree — one auto-runs a test on
hardware you did not intend, the other silently never runs it at all.
`test_moe_expert_parallel_2_gpu.py` is registered in both.

### 1.10 A bug that was avoided (worth recording)

`EP_PLAN.md` step 2 specified a sentinel `>= num_local_experts` for non-local
experts. That would have been silent memory corruption: `moe_align_block_size`
shifts ids by `+1` into an `(E+1)`-entry histogram, so an id equal to `E` writes
one slot past `shared_counts` and can later index `w1[E]` — an out-of-bounds
weight read.

The implementation used `-1` instead, which lands in the pad bucket and is srt's
own repo-wide sentinel. Verified at kernel level (`moe_align_kernel.cu`, the
Triton `fused_moe_kernel` filtered-block early-return, and
`token_dispatcher/standard.py`). The plan was wrong; the code was right.

---

## 2. Problems met

### 2.1 The output video differs from the dense path — the main investigation

The headline scare: baseline vs EP at 12 steps gave PSNR 13.8 dB, 98% of pixels
differing. That is far beyond bf16 rounding, and it looked like a correctness bug.

It is not. The resolution took four measurements:

1. **Noise floor.** Two identical baseline runs are **bitwise identical**. The
   pipeline is deterministic, so the comparison is a valid instrument and the
   delta is real signal.
2. **Isolated kernel.** Dense vs summed per-rank partials through the real
   `fused_experts` at production shape: cosine `0.9999933` = **0.94 bf16 ulp**,
   the floor for splitting a bf16 sum.
3. **Prediction and check.** 0.94 ulp x sqrt(48 blocks) predicts 2.53% after one
   DiT forward. Measured at 1 denoise step: **2.17%**, PSNR 31.0 dB. A prediction
   derived from (2) landing on an independent measurement is what closed this.
4. **Per-expert outputs are bitwise identical** to dense (`no_combine`
   comparison). The partition, the filtering and the slicing contribute exactly
   zero error.

The difference is only *where the top-k sum is rounded*: dense accumulates 8
contributions in fp32 and rounds once; EP rounds each rank's partial, then the
all-reduce rounds again. A 12-step sampler amplifies that chaotically into a
different — equally valid — draw.

Corroborating: the same test in **fp16** gives a max delta exactly **8x smaller**
(fp16 carries 3 more mantissa bits). The error tracks the ulp, which a logic bug
would not.

**fp32 does not fix it.** Measured three reduction strategies; an fp32 all-reduce
changes nothing (the partial was already rounded to bf16 *inside* `fused_experts`),
and reconstructing from per-slot outputs is slightly worse. Reconstructing dense
from **dense's own** per-slot outputs misses by the same margin — the kernel
computes `bf16(acc * w)` and nothing outside it can see anything but
`bf16(acc) * w`. Closing the gap needs an fp32 partial-output mode inside srt.

**Context that settles the policy question:** enabling FSDP with EP *off*
perturbs the sample **more** (cosine 0.808) than enabling EP does (0.968).
Sample-dependence on reduction order is pre-existing and already shipped.

### 2.2 `EP_PLAN.md`'s verification section is not achievable

The plan gated Phase 1 on final decoded-frame parity ("visually identical",
"~1e-2 relative on bf16"). Per 2.1 that can never pass for a correct bf16 EP
implementation at 12 steps. The usable gate is per-operator agreement plus the
1-step latent. `EP_PLAN.md` was left untracked rather than committed for this
reason; `expert_parallelism.mdx` carries the corrected guidance.

### 2.3 Disk: the checkpoint is 130 GB, not 60 GB

`robbyant/lingbot-video-moe-30b-a3b` ships a `refiner/` (13 shards, ~65 GB)
alongside `transformer/`. **The sglang pipeline never loads it** — `model_index.json`
lists only transformer / vae / text_encoder / processor / scheduler. A plain
`hf download` pulled 40 GB of pure waste and hit the pod's disk quota.

Worse, passing the **repo id** as `--model-path` makes the loader call a blanket
`snapshot_download` of all 63 files, re-pulling the refiner and blowing the quota
again mid-run. Workaround: download with `--exclude "refiner/*"` and pass the
**local snapshot path** as `--model-path`.

Possible follow-up: the loader could restrict its snapshot download to the
components the pipeline config actually declares.

### 2.4 Text encoder falls back on every run (pre-existing)

Every run logs a `ValueError: Following text encoder weights were not initialized
from checkpoint: ['lm_head.weight']` and falls back to the native HF version.
Unrelated to EP, identical in both arms, so it does not affect the comparison —
but it is a real pre-existing issue for this model, and it means the customized
text-encoder path is dead for LingBot-Video.

Note this is *not* the "Falling back to diffusers backend" gate from
`PROFILING_HANDOFF.md` — the DiT itself runs natively. That gate was checked and
holds.

### 2.5 Tooling friction

- `imageio`'s pyav plugin is broken in this env
  (`VideoCodecContext has no attribute 'close'`); decoded with PyAV directly.
- `python - <<EOF` heredocs fail on `import sglang` with a `CXXABI_1.3.15`
  sqlite/IPython error unless `LD_LIBRARY_PATH=/workspace/envs/sgl/lib` is
  exported, even though `python -m pytest` works fine.
- No `black` / `isort` / `ruff` in the env, so formatting was hand-checked
  against the 88-column convention rather than verified by tooling. **Worth
  re-running the real formatters before this leaves draft.**

---

## 3. How this was validated (reproduction)

Environment: 2x H100 80GB, `source /workspace/env.sh`. Two quirks: export
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` or `import sglang` fails inside a heredoc
with a `CXXABI_1.3.15` error, and pass the **local snapshot path** as
`--model-path` (a repo id makes the loader re-download all 63 files, refiner
included — see 2.3).

**Unit / kernel level**

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_lingbot_video_moe.py \
                 python/sglang/multimodal_gen/test/unit/test_fsdp_load.py -q   # 41 passed
python -m pytest python/sglang/multimodal_gen/test/single_test_file/test_moe_expert_parallel_2_gpu.py -q
```

The 2-GPU file is the one that matters for the partition: it asserts the ranks
hold *different* expert shards that tile `[0, num_experts)`, that a sharded
forward plus the real all-reduce matches a dense block (cosine 0.9999961,
~1 bf16 ulp), and that the router-replication guard fires when the ranks'
routers disagree.

**End-to-end.** Baseline and EP differ only in `--ep-size`:

```bash
MP=<local snapshot dir>
P="A red fox walks through a snowy forest at sunrise, camera slowly panning right"
COMMON="--num-gpus 2 --tp-size 2 --text-encoder-cpu-offload --dit-cpu-offload true \
        --width 640 --height 384 --num-frames 17 --num-inference-steps 12 \
        --seed 0 --save-output"

sglang generate --model-path "$MP" --prompt "$P" $COMMON --ep-size 1 --output-path out/base
sglang generate --model-path "$MP" --prompt "$P" $COMMON --ep-size 2 --output-path out/ep
```

Read latency from `Pixel data generated successfully in ... seconds`, per-step
from `[DenoisingStage] average time per step`, VRAM from `nvidia-smi` during the
run. Confirm the resolved config in the logged server-args JSON (`"ep_size": 2`)
— and confirm no `Falling back to diffusers backend` line, or the numbers are
void. Runs used for the tables:

| purpose | configuration |
| --- | --- |
| baseline | `--ep-size 1` (twice, for the noise floor) |
| EP | `--ep-size 2` (twice, for EP determinism) |
| 1-step | both arms at `--num-inference-steps 1` |
| FSDP arms | `--use-fsdp-inference true --dit-cpu-offload false`, `--ep-size` 1 and 2 |

**Comparing outputs.** Decode both videos with PyAV (imageio's pyav plugin is
broken in this env) and report MSE / PSNR / cosine / per-frame mean delta. The
throwaway script lived in the session scratchpad; it is ~30 lines around
`av.open(path)` + `frame.to_ndarray(format="rgb24")`.

**The reduction-strategy comparison** (bf16 vs fp32 vs `no_combine`) was a
standalone harness calling `fused_experts` directly at the production shape
(E=128, I=512, H=1024, top_k=8) under a single-rank srt process group, summing
per-rank partials in each strategy and reporting cosine against the dense call.
Not committed; the numbers it produced are in 2.1 and in the feature doc.

## 4. Known gaps in what was validated

- **Only `ep_size=2` was exercised on the real model.** `ep_size` 4 and 8 are
  covered synthetically (kernel level) but never on a full pipeline; this pod has
  2 GPUs. The *partition* is now covered at 2 ranks by
  `test_moe_expert_parallel_2_gpu.py`, but higher degrees remain single-process.
- **The FSDP+EP comparison changed two flags.** The FSDP runs also had
  `--dit-cpu-offload false`. Placement should not affect arithmetic, but the
  isolation is not clean.
- **Cross-node EP is untested** entirely.
- **`ep_size < tp_size` is unsupported** by construction (no orthogonal EP axis in
  `RankGenerator`).
- **The refit / weight-update path is EP-unaware** (`post_training/weights_updater.py`
  never calls `preprocess_loaded_state_dict`). This is pre-existing — it could not
  have worked for this model even before EP, since the w1+w3 packing is skipped —
  but EP adds a second reason.
- **`runtime/layers/lora/linear.py` re-shards a `base_layer` without
  `ignored_params`.** Harmless today (experts are raw `nn.Parameter`s, not
  LoRA-wrapped), but it is the one other `fully_shard` call site that would need
  the kwarg if LoRA ever wraps expert weights.
- ~~EP correctness rests on an unasserted invariant (router replication).~~
  **Closed**: the MoE block now checks router equality across the EP group on the
  first forward, and the 2-GPU test pins that the guard fires. The *input* side of
  the invariant (activations TP-identical, via `RowParallelLinear`) is still
  unasserted, but a violation there would break TP generally, not just EP.

---

## 5. Next steps

**Before leaving draft**

1. Run the real formatters (`black`, `isort`, `ruff`) — unavailable in this env.
2. Decide attribution/authorship on the commit before it becomes a PR.

**Upstream, separate from this PR**

3. File the two srt Triton bugs from 1.8. They are guarded here, not fixed, and
   they affect any EP user of the Triton MoE runner — not just diffusion.

**Feature work**

4. **Phase 2 — token dispatch.** This matters more than it first looks: the
   default `--num-gpus N` config puts every GPU on `sp_degree` with `tp_size=1`,
   where Phase 1 EP is unavailable by construction. Dispatch is what makes EP
   reachable in the configuration most users will actually run.
   `EP_PLAN.md` sketches 2a (pynccl `all_to_all_single`) then 2b (DeepEP).
5. **An fp32 partial-output mode in srt's `fused_experts`** — the only path to
   genuine parity with the dense sampler. Worth it only if sample-stability
   across `ep_size` turns out to matter to users; the doc currently takes the
   position that it does not, on the evidence that FSDP already perturbs more.
6. **Phase 3 — FP8/NVFP4 expert quantization**, per `EP_PLAN.md`. The load path
   would need to slice per-expert *scales* on the same expert range as the
   weights.
7. ~~Assert the router-replication invariant.~~ Done — see section 4.
