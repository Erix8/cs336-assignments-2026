# ⚙️ CS336 Assignment 1 — Basics: A Transformer LM, Built by Hand

The from-scratch rule is strict: no `torch.nn.Linear`, no `F.cross_entropy`, no `torch.optim.AdamW` — only
`torch.nn.Parameter`, container classes, and the `torch.optim.Optimizer` base class. 🧱 The target is a
~17M-parameter decoder-only Transformer that writes (mostly) coherent TinyStories 🧒, then a 45-minute
OpenWebText run 🌐 for the course leaderboard. Handout: [`cs336_assignment1_basics.pdf`](./cs336_assignment1_basics.pdf).

**Progress** — ⬜⬜⬜⬜⬜⬜⬜ **0 / 7 pieces** written (105 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **BPE tokenizer** · 38 pts — train merges on TinyStories (10K vocab) and OpenWebText (32K), then
  `encode` / `encode_iterable` / `decode` with special tokens, plus a compression-ratio/tokenizer study ⚙️
- [ ] **Transformer LM** · 29 pts — Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, scaled dot-product
  attention, causal multi-head self-attention, pre-norm block, output head 🕸️
- [ ] **Loss & optimizer** · 8 pts — cross-entropy, AdamW from scratch, cosine LR schedule with warmup,
  gradient clipping 📉
- [ ] **Training loop** · 7 pts — `np.memmap` data loader, checkpoint save/load, configurable training script 🔁
- [ ] **Decoding** · 3 pts — prompt completion with temperature scaling and top-p (nucleus) sampling 🎲
- [ ] **Experiments** · 12 pts — LR sweep, batch-size study, ablations (RMSNorm off, post-norm, NoPE,
  SwiGLU vs SiLU) 🧪
- [ ] **OpenWebText + leaderboard** · 8 pts — full OWT run, then a ≤45-min B200 submission 🏁

## 🧐 What I want to understand

- Why **bytes → merges → tokens** works at all, and what the tokenizer silently decides for the model 🫥
- Numerical stability as craft: max-subtraction in softmax, ε in AdamW, `M / (‖g‖ + ε)` in clipping 🧯
- The pre-norm residual highway `x + f(Norm(x))` — and why it displaced the original post-norm block 🛣️
- Whether causal attention alone encodes position well enough that **NoPE** competes with RoPE 🌀
- Where the FLOPs, memory and wall-clock of a 17M-param model actually go ⏱️

## 📝 Field notes

A running log — deliberately empty for now, it fills up as the pieces land.

- **Status quo:** every test fails with a proud `NotImplementedError`, and `cs336_basics/` is empty by design.
- **House rules:** build it from scratch; get it correct on CPU 🐢 before fast on GPU 🚀; gradcheck every
  hand-written gradient against a reference before trusting it.
- **Debug loop:** overfit a **single minibatch** first → inspect tensor shapes → watch activation/weight/gradient
  norms for vanishing or exploding 📈
- **Gotcha I'll remember:** on `mps`, skip TF32 kernels — they're silently broken there 🚫

## ✅ Definition of done

- Tokenizer round-trips match `tiktoken` byte-for-byte (ASCII, unicode, special tokens) 🔁
- All test files green 🟢
- TinyStories LM: ~17M params trained on 327.68M tokens, **val loss ≤ 1.45**
- Generated samples: at least 256 tokens of fluent TinyStories-style text 🧒
- Ablation curves (RMSNorm / post-norm / NoPE / SiLU) with commentary 🔬
- OpenWebText run: loss curve + generated text 🌐
- Leaderboard entry: **< 5.0** val loss inside a 45-min B200 run 🏆

## 🏃 Getting it running

1. `uv sync` — the environment is managed with [`uv`](https://docs.astral.sh/uv/) 🐍
2. Fetch the data (**not** committed here, ≈14 GB):
   ```sh
   mkdir -p data && cd data
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
   wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
   wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
   cd ..
   ```
3. `uv run pytest` — red everywhere at first, which is exactly right 🩸
4. Implement inside `cs336_basics/`, wire it into [`tests/adapters.py`](./tests/adapters.py) (glue only), re-run 🔁

> 💡 Correctness on CPU, speed on GPU — and `make_submission.sh` does the Gradescope packaging 📬

## 🧭 Repo layout

- `cs336_basics/` — my implementation (empty right now, on purpose)
- `tests/adapters.py` — glue between my code and the tests
- `tests/test_*.py` — the suites I have to pass: `test_train_bpe`, `test_tokenizer`, `test_model`, `test_nn_utils`,
  `test_optimizer`, `test_data`, `test_serialization`
- `data/` — TinyStories + OpenWebText (downloaded, gitignored)
- `pyproject.toml` · `make_submission.sh` · `cs336_assignment1_basics.pdf`

---

First token, then the world. 🚀

