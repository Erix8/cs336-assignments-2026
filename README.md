# 🎯 CS336 Assignments — Spring 2026 · My Solutions 🛠️

My solutions to [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu) 🌱 — building
every piece of an LM by hand: tokenizer, Transformer, optimizer, kernels, parallelism, data, and alignment. 💪

## 🗂️ Assignment Contents

| # | Assignment | Summary | Progress |
|---|------------|---------|----------|
| 1 | [⚙️ Basics](./assignment1-basics/README.md) | BPE tokenizer, Transformer LM (RMSNorm, RoPE, SwiGLU, MHA), and the training loop (AdamW, cosine LR, grad clipping) — TinyStories / OWT main runs, LR and batch-size studies, four ablations, and a 32-page report 📚 | 🟢 **Current scope complete** · paused · leaderboard deferred |
| 2 | [🏭 Systems](./assignment2-systems/README.md) | Profile & benchmark A1, write FlashAttention-2 in Triton ⚡, add DDP / FSDP / sharded optimizer 🧩 | ⬜ **Not Started** |
| 3 | [📈 Scaling](./assignment3-scaling/README.md) | Sweep Transformer components via a training API and fit a scaling law 📉 | ⬜ **Not Started** |
| 4 | [📖 Data](./assignment4-data/README.md) | Common Crawl → clean text: language ID, PII masking, quality/toxicity filtering, deduplication 🧹 | ⬜ **Not Started** |
| 5 | [🤝 Alignment](./assignment5-alignment/README.md) | SFT + GRPO / DPO to teach an LM to reason, then evaluate on GSM8K / MMLU 🧮 | ⬜ **Not Started** |

## 📌 Current progress

**Assignment 1** is wrapped up for now (2026-10-18):

- ✅ Implementation complete: BPE tokenizer, Transformer, optimizer, training loop, and decoding.
- ✅ Low-resource experiments complete: LR study, four architecture ablations, batch sizes 1 / 8 / 16 / 32 / 48, and TinyStories / OWT main runs.
- ✅ Standard-budget main runs complete (327.68M tokens each): TinyStories final validation loss **1.3245** (best sampled **1.3240**), meeting the ≤ 1.45 target; OWT **3.8596**.
- 📄 The **32-page report** includes results, protocol caveats, diagnostics, and generation samples. See the [assignment README](./assignment1-basics/README.md) and [report source](./assignment1-basics/report/main.tex).
- ⏸️ Leaderboard training/submission remains deferred; no further A1 experiments are planned at this stage.

Assignments 2–5 have not been started.

## 🚀 Setup

### VS Code

Open [`cs336-assignments.code-workspace`](./cs336-assignments.code-workspace) instead of the repository root as a single folder. The multi-root workspace treats every assignment as an independent Python project and selects that assignment's `.venv`.

```sh
code cs336-assignments.code-workspace
```

Each folder is its own [`uv`](https://docs.astral.sh/uv/) project:

```sh
cd assignment1-basics
uv sync
uv run pytest
```

- 🔑 **A3**: `export A3_API_KEY=<your 8-digit student ID>` before using the training API.
- 🤗 **A5**: install in two steps — `uv sync --no-install-package flash-attn && uv sync`.
- 📦 Submit with each folder's `make_submission.sh` / `test_and_make_submission.sh`.
