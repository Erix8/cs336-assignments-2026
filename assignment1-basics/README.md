# ⚙️ CS336 Assignment 1 — Basics: A Transformer LM, Built by Hand

The from-scratch rule is strict: no `torch.nn.Linear`, no `F.cross_entropy`, no `torch.optim.AdamW` — only
`torch.nn.Parameter`, container classes, and the `torch.optim.Optimizer` base class. 🧱 The reference model is a
22.7M-parameter decoder-only Transformer (10K vocab, 4 layers, width 512, 16 heads, SwiGLU 1344) trained on
TinyStories 🧒; OWT uses a 32K vocabulary and 45.2M parameters. Both standard-budget main runs are complete; leaderboard training/submission remains deferred. Handout:
[`cs336_assignment1_basics.pdf`](./cs336_assignment1_basics.pdf).

**Status (2026-10-18)** — Current study scope complete; paused for now. Implementation, low-resource
experiments, standard-budget TinyStories / OWT main runs, and the 32-page report are complete.
Leaderboard training/submission has not been attempted. No further runs are planned at this stage.

| Piece | Status | Evidence |
| ----- | ------ | -------- |
| BPE tokenizer | ✅ | 10K TinyStories and 32K OWT vocabularies; streaming encoding, decoding, special tokens |
| Transformer LM | ✅ | Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, pre/post-norm block, LM head |
| Loss & optimizer | ✅ | cross-entropy, SGD, AdamW, cosine warmup/decay, gradient clipping |
| Training loop | ✅ | memory-mapped `uint16` data, checkpoint/resume, W&B logging, fixed-window evaluation |
| Decoding | ✅ | temperature + top-p sampling; EOS/length stopping and token counts |
| Experiments | ✅ | LR study, four ablations, batch-size study, low-resource and standard-budget main runs |
| Report | ✅ | [Final PDF](./writeup.pdf); protocols, diagnostics, plots, and full samples |
| Leaderboard | ⏸️ | no leaderboard-specific run or submission; no prescribed wall-clock-budget result claimed |

## 📊 Completed results

The shared backbone has context length 256, four layers of width 512, 16 heads, SwiGLU width 1344,
pre-RMSNorm, and RoPE. All experiments use seed 42. TinyStories uses 10K vocabulary / 22.7M parameters;
OWT uses 32K / 45.2M, so their losses and compute are not directly comparable.

| Dataset | Training tokens | Device | Final val loss | Best sampled val loss | Logged time (min) |
| ------- | --------------- | ------ | -------------- | --------------------- | ----------------- |
| TinyStories | 40.96M | M4 / MPS | 1.5879 | 1.5879 | 87.03 |
| OWT | 40.96M | RTX 4090D / CUDA | 4.4561 | 4.4561 | ~10.5 |
| TinyStories | 327.68M | RTX 4090D / CUDA | **1.3245** | **1.3240** at update 79,200 | 66.85 |
| OWT | 327.68M | RTX 4090D / CUDA | **3.8596** | **3.8596** at update 80,000 | 79.45 |

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

3. Encode TinyStories using the committed tokenizer (repeat with `owt` for OWT), then run tests:

   ```sh
   uv run python -m cs336_basics.tokenize_corpus tinystories train
   uv run python -m cs336_basics.tokenize_corpus tinystories valid
   uv run pytest
   ```

   Historical checks: the original suite had 46 passes / 2 skips; separate added suites had 61 passes.
   These are earlier checks, not a newly rerun full-suite result.

4. Train the reference TinyStories model (≈87 min on M4 MPS):

   ```sh
   uv run python -m cs336_basics.train \
     --train-path data/tokenized/tinystories_train_uint16.bin \
     --valid-path data/tokenized/tinystories_valid_uint16.bin \
     --vocab-size 10000 --device mps --seed 42 \
     --batch-size 16 --max-steps 10000 \
     --max-learning-rate 1e-3 --min-learning-rate 1e-4 --warmup-iters 100 \
     --eval-seed 42 --eval-batch-size 16 --eval-batches 20 \
     --eval-interval 100 --save-interval 1000 \
     --out-dir checkpoints/readme-baseline \
     --wandb-project cs336-assignment1
   ```

   For the standard-budget recipe, use `--device cuda --max-steps 80000 --warmup-iters 800`
   and `--eval-interval 800 --save-interval 8000`, with a fresh output directory. For OWT, also
   change both data paths to `owt_*_uint16.bin` and use `--vocab-size 32000`.

5. Generate from the checkpoint produced above:

   ```sh
   uv run python -m cs336_basics.generate_text \
     --checkpoint checkpoints/readme-baseline/final.pt \
     --vocab-file data/tokenizers/tinystories_train_vocab.json \
     --merges-file data/tokenizers/tinystories_train_merges.json \
     --prompt "Once upon a time" --max-new-tokens 256 \
     --temperature 0.7 --top-p 0.9 --device mps --seed 42
   ```

> 💡 Raw corpora, tokenized `uint16` arrays, and checkpoints are gitignored; the JSON tokenizer artifacts under
> `data/tokenizers/` are committed. `make_submission.sh` handles Gradescope packaging 📬

## 🧭 Repo layout

📄 **[Read the final report](./writeup.pdf)** — 32 pages of derivations, experiments, and generation samples.

```text
assignment1-basics/
├── writeup.pdf                   Final report
├── cs336_assignment1_basics.pdf   Assignment handout
├── cs336_basics/
│   ├── model.py                  Transformer layers and language model
│   ├── tokenizer.py              BPE encoding and decoding
│   ├── train_bpe.py              BPE vocabulary training
│   ├── nn_utils.py               Loss, softmax, and gradient clipping
│   ├── optimizer.py              SGD, AdamW, and LR schedule
│   ├── data.py                   Memory-mapped token sampling
│   ├── serialization.py          Checkpoint save/load
│   ├── train.py                  Training CLI
│   ├── generate_text.py          Sampling CLI
│   ├── tokenize_corpus.py        Corpus-to-uint16 pipeline
│   ├── *_experiments.py          Tokenizer and optimizer experiments
│   └── transformer_accounting.py Parameter and FLOP accounting
├── tests/                        Provided tests, adapters, and added tests
├── report/                       LaTeX source, appendices, and figures
├── data/
│   ├── tokenizers/               Tracked vocabulary and merge JSONs
│   └── tokenized/                Local uint16 arrays
├── checkpoints/                  Local training runs, logs, and samples
├── pyproject.toml                Project dependencies and configuration
└── make_submission.sh            Submission packaging
```

Raw corpora also live in `data/`. Corpora, tokenized arrays, and checkpoints are gitignored.

---

First token, then the world. 🚀
