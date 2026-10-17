# 🎯 CS336 Assignments — Spring 2026 · My Solutions 🛠️

My solutions to [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu) 🌱 — building
every piece of an LM by hand: tokenizer, Transformer, optimizer, kernels, parallelism, data, and alignment. 💪

## 🗂️ Assignment Contents

| # | Assignment | Summary | Progress |
|---|------------|---------|----------|
| 1 | [⚙️ Basics](./assignment1-basics/README.md) | BPE tokenizer, Transformer LM (RMSNorm, RoPE, SwiGLU, MHA), and the training loop (AdamW, cosine LR, grad clipping) — a 22.7M-param TinyStories LM, LR study, four ablations, and a 24-page report 📚 | 🟢 **Core complete** · batch-size study running · OWT/leaderboard deferred |
| 2 | [🏭 Systems](./assignment2-systems/README.md) | Profile & benchmark A1, write FlashAttention-2 in Triton ⚡, add DDP / FSDP / sharded optimizer 🧩 | ⬜ **Not Started** |
| 3 | [📈 Scaling](./assignment3-scaling/README.md) | Sweep Transformer components via a training API and fit a scaling law 📉 | ⬜ **Not Started** |
| 4 | [📖 Data](./assignment4-data/README.md) | Common Crawl → clean text: language ID, PII masking, quality/toxicity filtering, deduplication 🧹 | ⬜ **Not Started** |
| 5 | [🤝 Alignment](./assignment5-alignment/README.md) | SFT + GRPO / DPO to teach an LM to reason, then evaluate on GSM8K / MMLU 🧮 | ⬜ **Not Started** |

## 📌 Current progress

**Assignment 1** is functionally complete at the adopted low-resource scope:

- ✅ All implementation sections done — tokenizer, Transformer, optimizer, training loop, and decoding; `87 passed, 2 skipped`.
- ✅ TinyStories reference run (40.96M tokens): **val loss 1.5879**, below the adopted ≤ 2.00 target.
- ✅ LR study: 1e-3 was best at 1.5966; peaks ≥ 0.3 showed repeated finite loss explosions.
- ✅ Four ablations: no-RMSNorm 1.6598, post-norm 1.6042, NoPE 1.7010, SiLU 1.6106 (single seed).
- 🟨 Batch-size study: batch 48 finished at 1.6329; batch 1 / 8 / 32 launched, analysis pending.
- ⏸️ OpenWebText and the leaderboard submission are deferred pending suitable GPU access.
- 📄 Full results, caveats, and reproduction commands: [assignment1-basics/README.md](./assignment1-basics/README.md) and [`report/main.tex`](./assignment1-basics/report/main.tex).

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
