# 🏭 CS336 Assignment 2 — Systems: Making Transformers Fast

Make the Assignment-1 model fast on one GPU, then spread it across many. 🔧 First profile and benchmark
where time and memory actually go, then write **FlashAttention-2 in Triton** ⚡, then climb the parallelism
ladder: DDP → optimizer-state sharding → FSDP, backing every step with hand-derived communication math.
Training on the staff A1 model in [`cs336-basics/`](./cs336-basics). Handout:
[`cs336_assignment2_systems.pdf`](./cs336_assignment2_systems.pdf).

**Progress** — ⬜⬜⬜⬜⬜⬜⬜⬜ **0 / 8 pieces** written (137 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **Profiling & benchmarking harness** · 16 pts — time fwd/bwd/optimizer step, Nsight Systems trace,
  fp16/bf16 mixed precision (+ accumulation), memory profiling 📊
- [ ] **Activation checkpointing** · 4 pts — recompute activations to trade compute for memory 🧮
- [ ] **Attention kernels** · 29 pts — PyTorch baseline, `torch.compile`, then **FlashAttention-2** in Triton
  (forward + backward), benchmarked at batch 1 ⚡
- [ ] **DDP from scratch** · 21 pts — ring all-reduce benchmarks, a naive DDP, then overlapping
  per-parameter gradient sync 🔗
- [ ] **Optimizer state sharding** · 20 pts — ZeRO-style sharding of the optimizer moments 💾
- [ ] **FSDP** · 20 pts — all-gather weights for compute, reduce-scatter gradients 🧩
- [ ] **Parallelism math** · 17 pts — alternate ring all-reduce, plus DP / FSDP / TP / 2D calculations ✍️
- [ ] **Leaderboard** · 10 pts — fastest training step on 2× B200 🏁

## 🧐 What I want to understand

- The **iron law** — compute, memory traffic and communication as three budgets to trade off ⚖️
- Why CUDA calls are async, and why `torch.cuda.synchronize()` is non-negotiable when timing ⏱️
- Tiling + online softmax: how FlashAttention keeps attention inside SRAM without materializing `N×N` 🧠
- Overlapping all-reduce with the backward pass instead of paying for it at the end 🔀
- How far sharding goes: optimizer states → gradients → parameters (ZeRO stages, FSDP) 🪜

## 📝 Field notes

A running log — deliberately empty for now, it fills up as the pieces land.

- **Measure first:** warm-up iterations before timing; NCCL and Nsight both need them.
- **Two backends:** debug locally with Gloo on CPU, benchmark for real with NCCL on GPU.
- **Across ranks:** aggregate timings with `dist.all_gather_object` — per-rank numbers wobble.
- **Leaderboard gotcha:** from a cold PyTorch/Triton cache the run must finish in 10 minutes, so be careful
  with aggressive `torch.compile` and Triton autotuning. 🥶
- **Model sizes to test:** small → medium → large → xl → 10B (mostly GPT-2 configs, ctx 512, batch 4).

## ✅ Definition of done

- FlashAttention-2 (Triton) matches the PyTorch reference in forward **and** backward ⚡
- All test suites green: `test_attention`, `test_ddp`, `test_fsdp`, `test_sharded_optimizer` 🟢
- Benchmark tables/plots for the model sizes, with commentary on time and memory 📊
- DDP / optimizer-sharding / FSDP run correctly across ≥2 GPUs 🧩
- Leaderboard: one full training step (batch 2, 2× B200) well under the **10 s** baseline 🏆

## 🏃 Getting it running

1. `uv sync` — deps are managed with [`uv`](https://docs.astral.sh/uv/); the local `cs336-basics` package is
   pulled in via `pyproject.toml`.
2. Sanity-check the import: `uv run python -c "import cs336_basics.model"`.
3. Wire my code into [`tests/adapters.py`](./tests/adapters.py) (glue only) and run `uv run pytest` — the
   DDP/FSDP tests need ≥2 GPUs, so debug on CPU + Gloo first 🔁
4. Package for Gradescope with [`test_and_make_submission.sh`](./test_and_make_submission.sh) 📬

> 💡 Want my *own* A1? Point `pyproject.toml` at my `assignment1-basics` instead of the staff `cs336-basics`.

## 🧭 Repo layout

- `cs336_systems/` — my implementation (empty right now, on purpose)
- `cs336-basics/` — staff Assignment-1 model to profile (`cs336_basics.model`, …)
- `tests/adapters.py` — glue between my code and the tests
- `tests/test_attention.py` · `test_ddp.py` · `test_fsdp.py` · `test_sharded_optimizer.py` — the suites to pass
- `pyproject.toml` · `test_and_make_submission.sh` · `cs336_assignment2_systems.pdf`

---

Fast on one GPU, then fast on all of them. ⚡

