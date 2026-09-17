# 🎯 CS336 Assignments — Spring 2026 · My Solutions 🛠️

My solutions to [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu) 🌱 — building
every piece of an LM by hand: tokenizer, Transformer, optimizer, kernels, parallelism, data, and alignment. 💪

## 🗂️ Assignment Contents

| # | Assignment | Summary | Progress |
|---|------------|---------|----------|
| 1 | [⚙️ Basics](./assignment1-basics/README.md) | BPE tokenizer, Transformer LM (RMSNorm, RoPE, SwiGLU, MHA), and the training loop (AdamW, cosine LR, grad clipping) — train a small LM 📚 | ⬜ **Not Started** |
| 2 | [🏭 Systems](./assignment2-systems/README.md) | Profile & benchmark A1, write FlashAttention-2 in Triton ⚡, add DDP / FSDP / sharded optimizer 🧩 | ⬜ **Not Started** |
| 3 | [📈 Scaling](./assignment3-scaling/README.md) | Sweep Transformer components via a training API and fit a scaling law 📉 | ⬜ **Not Started** |
| 4 | [📖 Data](./assignment4-data/README.md) | Common Crawl → clean text: language ID, PII masking, quality/toxicity filtering, deduplication 🧹 | ⬜ **Not Started** |
| 5 | [🤝 Alignment](./assignment5-alignment/README.md) | SFT + GRPO / DPO to teach an LM to reason, then evaluate on GSM8K / MMLU 🧮 | ⬜ **Not Started** |

## 🚀 Setup

Each folder is its own [`uv`](https://docs.astral.sh/uv/) project:

```sh
cd assignment1-basics
uv sync
uv run pytest
```

- 🔑 **A3**: `export A3_API_KEY=<your 8-digit student ID>` before using the training API.
- 🤗 **A5**: install in two steps — `uv sync --no-install-package flash-attn && uv sync`.
- 📦 Submit with each folder's `make_submission.sh` / `test_and_make_submission.sh`.
