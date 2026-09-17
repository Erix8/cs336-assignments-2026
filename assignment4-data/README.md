# 📖 CS336 Assignment 4 — Data: Turning Common Crawl into Training Data

A raw web crawl is not training data — it's HTML, boilerplate, duplicated boilerplate, and worse. 🕸️ The job
is to build the pipeline that fixes that: extract text, filter by language, mask PII, drop harmful and
low-quality pages, deduplicate — then train a model on my own filtered corpus and see whether it beats the
baseline perplexity on the C4-100-domains slice of Paloma. Handout:
[`cs336_assignment4_data.pdf`](./cs336_assignment4_data.pdf).

**Progress** — ⬜⬜⬜⬜⬜⬜⬜⬜ **0 / 8 pieces** written (71 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **Data reconnaissance** · 4 pts — read a raw WARC and its WET file, annotate ~25 documents 👀
- [ ] **HTML → text** · 3 pts — Resiliparse extraction with encoding detection 🌐
- [ ] **Language filtering** · 6 pts — fastText `lid.176.bin` language ID 🗣️
- [ ] **PII masking** · 3 pts — find and mask emails, phone numbers and IPs 🕵️
- [ ] **Harmful-content filtering** · 6 pts — Dolma fastText NSFW + hate-speech classifiers ☣️
- [ ] **Quality filtering** · 18 pts — Gopher heuristic rules **plus** a trained fastText quality classifier 🐹
- [ ] **Deduplication** · 11 pts — exact line-level, then MinHash + LSH for fuzzy document duplicates 🧹
- [ ] **Filter → inspect → tokenize → train** · 20 pts — apply the pipeline, eyeball what got dropped,
  serialize `uint16` token IDs, and train a GPT-2-small-shaped model 🏋️

## 🧐 What I want to understand

- WARC vs. WAT vs. WET, and why "the main content" is so hard to isolate from HTML
- The precision/recall dial on every filter — each one quietly shapes what the model learns ⚖️
- Why dedup pays off so well: repeated boilerplate just burns the compute budget 📉
- MinHash + LSH: approximating Jaccard similarity without comparing every pair 🔎
- How much of a perplexity win comes from *data* alone, with the architecture frozen 🧪

## 📝 Field notes

A running log — deliberately empty for now, it fills up as the pipeline runs.

- **Never leak validation:** Paloma C4-100 may inform my filters, but no validation text may enter training data.
- **Parallelize early:** these WET files are huge — reach for `concurrent.futures` / `multiprocessing` from the start.
- **Everything else is frozen:** don't touch the model or the training script; the whole point is the data.
- **Freeze before a 2-hour run:** a bad filter is expensive to discover after the GPU budget is gone. 💸

## ✅ Definition of done

- All `tests/test_*.py` suites green (`extract`, `langid`, `pii`, `quality`, `toxicity`, `deduplication`) 🟢
- A filtered + deduplicated corpus, tokenized to `uint16` with `<|endoftext|>` between documents 🔤
- GPT-2-small-shaped (~430M params) trained: 8× B200, batch 128/device, 16,384 steps, ctx 512 (~8.6B tokens)
- Validation perplexity on Paloma C4 100 domains better than the unfiltered baseline 🏆

## 🏃 Getting it running

1. `uv sync` — deps are managed with [`uv`](https://docs.astral.sh/uv/), and Modal sponsors the compute ☁️
2. Get the data (students read from `/shared-data`): `uv run scripts/download_data.py --offline-only` for the
   offline subset, or `uv run modal run scripts/download_data.py` for everything — the latter only after
   implementing `is_english` in `cs336_data/wet_files.py` 📥
3. Launch training on my tokenized shard:
   `uv run modal run scripts/train.py --train-bin /root/data/your_data.bin`

## 🧭 Repo layout

- `cs336_data/` — my pipeline: extraction, filtering, PII, classifiers, dedup (mostly `TODO`s) ✍️
- `cs336_basics/` — staff trainer from Assignment 1 (~430M-param GPT-2 small); do **not** modify it 🔒
- `scripts/download_data.py` · `scripts/train.py` · `scripts/generate_with_gpt2_tok.py` — data + training entrypoints
- `tests/adapters.py` + `tests/test_*.py` — the suites to pass
- `configs/` · `pyproject.toml` · `test_and_make_submission.sh` · `cs336_assignment4_data.pdf`

---

Better data beats a better model. 🧹

