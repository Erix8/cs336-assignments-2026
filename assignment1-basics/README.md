# ⚙️ CS336 Assignment 1 — Basics: A Transformer LM, Built by Hand

The from-scratch rule is strict: no `torch.nn.Linear`, no `F.cross_entropy`, no `torch.optim.AdamW` — only
`torch.nn.Parameter`, container classes, and the `torch.optim.Optimizer` base class. 🧱 The reference model is a
22.7M-parameter decoder-only Transformer (10K vocab, 4 layers, width 512, 16 heads, SwiGLU 1344) trained on
TinyStories 🧒. The OpenWebText leaderboard run is deferred until suitable GPU access is available. Handout:
[`cs336_assignment1_basics.pdf`](./cs336_assignment1_basics.pdf).

**Progress** — 🟩🟩🟩🟩🟩🟨⬜ **5 / 7 pieces complete · 1 in progress · 1 deferred** · `87 passed, 2 skipped` ⚙️

| Piece | Status | Evidence |
| ----- | ------ | -------- |
| BPE tokenizer | ✅ | 10K TinyStories and 32K OWT vocabularies; `encode` / `encode_iterable` / `decode` with special tokens |
| Transformer LM | ✅ | from-scratch Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, attention, pre/post-norm block, LM head |
| Loss & optimizer | ✅ | cross-entropy, SGD, AdamW, cosine schedule with warmup, global-norm gradient clipping |
| Training loop | ✅ | `np.memmap` `uint16` data loader, checkpoint/resume, W&B logging |
| Decoding | ✅ | temperature + top-p sampling CLI; 256-token TinyStories samples in the report |
| Experiments | 🟨 | LR sweep and four ablations complete; batch-size study running |
| OWT + leaderboard | ⏸️ | explicitly deferred pending GPU resources |

## 📊 Results so far

All runs use TinyStories, a 10K vocabulary, context length 256, four layers of width 512, 16 heads, and seed 42
on Apple M4 / MPS. The low-resource budget is **40.96M tokens** (batch 16 × 10,000 updates), and the adopted
target is **validation loss ≤ 2.00**. The handout's larger 327.68M-token budget and its 1.45 target were not
attempted. Unless noted, validation averages 20 fixed batches of 16 windows (81,920 tokens) from a private
seed-42 generator; the older LR sweep used random validation batches, so do not compare across protocols.

| Run | Budget | Peak LR | Final val loss | Note |
| --- | ------ | ------- | -------------- | ---- |
| Matched baseline | 40.96M | 1e-3 | **1.5879** | pre-RMSNorm + RoPE + SwiGLU |
| LR sweep | 40.96M | 5e-4 | 1.6402 | random validation batches |
| LR sweep | 40.96M | 1e-3 | 1.5966 | selected peak |
| LR sweep | 40.96M | 2e-3 | 1.6092 | random validation batches |
| No RMSNorm | 40.96M | 5e-4 | 1.6598 | reduced LR after 1e-3 instability |
| Post-norm | 40.96M | 1e-3 | 1.6042 | final model norm retained |
| NoPE | 40.96M | 1e-3 | 1.7010 | causal mask retained |
| SiLU FFN | 40.96M | 1e-3 | 1.6106 | width 2048 to approximately match parameters |
| Batch 48 | 40.96M | 1e-3 | 1.6329 | 3,333 updates instead of 10,000 |
| Batch 1 / 8 / 32 | 40.96M each | 1e-3 | — | launched; results pending |

The best TinyStories run reached **1.5879**, comfortably below the low-resource target. These are single-seed
comparisons, so the small gaps (for example post-norm at +0.0163) are not significance claims.

## 🧐 What the experiments showed

- **Normalization buys stability at a higher LR.** Removing RMSNorm at 1e-3 produced a transient loss
  explosion (val 2125.22 at update 400) before recovering; 5e-4 trained cleanly but ended 0.0719
  worse than the baseline. This gap mixes architecture and LR changes 🧯
- **RoPE still matters at this scale.** NoPE had the largest degradation (+0.1131), so causal attention alone did
  not replace positional information in this 4-layer model 🌀
- **Pre-norm vs. post-norm is a small effect here.** Post-norm trained fine and lost by 0.0163; one seed cannot
  establish significance 🛣️
- **Tokenizer choice is a real cost.** Reusing the TinyStories tokenizer on OWT needs about a third more tokens,
  which directly reduces effective context and compute 🫥
- **Open questions:** batch-size effects at equal tokens, and whether the ordering survives multi-seed runs or a
  larger token budget ⏱️

## 🏃 Getting it running

1. `uv sync` — the environment is managed with [`uv`](https://docs.astral.sh/uv/) 🐍
2. Fetch the raw corpora (**not** committed, ≈14 GB):

   ```sh
   mkdir -p data && cd data
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
   wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
   wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
   cd ..
   ```

3. `uv run pytest` — should print `87 passed, 2 skipped` 🟢
4. Train the reference TinyStories model (≈87 min on M4 MPS):

   ```sh
   uv run python -m cs336_basics.train \
     --train-path data/tokenized/tinystories_train_uint16.bin \
     --valid-path data/tokenized/tinystories_valid_uint16.bin \
     --vocab-size 10000 --device mps --seed 42 \
     --eval-seed 42 --out-dir checkpoints/readme-baseline\ 
     --wandb-project cs336-assignment1
   ```

5. Generate from a checkpoint:

   ```sh
   uv run python -m cs336_basics.generate \
     --checkpoint checkpoints/baseline/final.pt \
     --vocab-file data/tokenizers/tinystories_train_vocab.json \
     --merges-file data/tokenizers/tinystories_train_merges.json \
     --prompt "Once upon a time" --max-new-tokens 256 \
     --temperature 0.7 --top-p 0.9 --device mps --seed 42
   ```

> 💡 Raw corpora, tokenized `uint16` arrays, and checkpoints are gitignored; the JSON tokenizer artifacts under
> `data/tokenizers/` are committed. `make_submission.sh` handles Gradescope packaging 📬

## 🧭 Repo layout

- `cs336_basics/model.py` — Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, Transformer block and LM
- `cs336_basics/tokenizer.py` · `train_bpe.py` — BPE training, encoding, decoding, special tokens
- `cs336_basics/nn_utils.py` · `optimizer.py` — loss, softmax, clipping, SGD, AdamW, cosine schedule
- `cs336_basics/data.py` · `serialization.py` · `train.py` · `generate.py` — data loader, checkpoints, training and
  decoding entry points
- `cs336_basics/tokenize_corpus.py` · `bpe_experiments.py` · `tokenizer_experiments.py` — corpus-to-`uint16`
  pipeline, BPE training driver, and tokenizer measurements
- `cs336_basics/transformer_accounting.py` · `optimizer_experiments.py` — parameter/FLOP accounting
- `tests/adapters.py` — glue between the implementation and the provided tests; added tests live in separate
  `test_*_properties.py` / `test_*_edge_cases.py` files so the originals stay untouched
- `report/` — 24-page LaTeX write-up (`main.tex`, `sections/`, W&B figure exports); build with
  `cd report && latexmk -pdf main.tex`
- `data/` — raw corpora (gitignored), tokenizer JSONs (tracked), tokenized arrays (gitignored)
- `checkpoints/` — LR sweep, ablations, batch-size runs (gitignored)
- `pyproject.toml` · `make_submission.sh` · `cs336_assignment1_basics.pdf`

---

First token, then the world. 🚀
