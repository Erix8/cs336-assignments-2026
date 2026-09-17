# 🤝 CS336 Assignment 5 — Alignment: Reasoning RL

Pre-training gave us a base model; now we teach it to *solve* something. 🧠 Start with prompting baselines
(zero-shot, few-shot, chain-of-thought) on GSM8K, then implement **GRPO** from scratch and run it on
OLMo-2-0425-1B with a vLLM rollout server, and finish by exploring the contested corners of policy-gradient
estimation: Dr. GRPO, RFT, MaxRL, and off-policy importance reweighting. Handout:
[`cs336_spring2026_assignment5_alignment.pdf`](./cs336_spring2026_assignment5_alignment.pdf) — plus the
[optional safety/RLHF supplement](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf).

**Progress** — ⬜⬜⬜⬜⬜⬜⬜⬜ **0 / 8 pieces** written (90 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **Prompting baselines** · 5 pts — `question_only` vs. `r1_zero` vs. `r1_zero_three_shot` on GSM8K 📝
- [ ] **Policy-gradient theory** · 5 pts — derive and estimate the variance of the PG estimator ✍️
- [ ] **GRPO components** · 5.5 pts — prompt/output tokenization, response log-probs + entropy, rollout
  rewards, group-normalized advantages, PG loss, microbatch aggregation 🧩
- [ ] **On-policy GRPO** · 15 pts — the full training step plus the training loop on OLMo-2-0425-1B 🤖
- [ ] **Tuning & ablations** · 6 pts — learning-rate sweep and a prompt ablation 📊
- [ ] **RL variants** · 25 pts — Dr. GRPO (drop std / length normalization), RFT, MaxRL, plus the
  difficulty-reweighting derivations 🎛️
- [ ] **Off-policy RL** · 18.5 pts — importance reweighting, PPO/GSPO-style clipping, token-level ratios ⚖️
- [ ] **My own estimator** · 10 pts — design a policy-gradient estimator and show whether it helps 🧪

## 🧐 What I want to understand

- Why optimizing `J(θ) = E[r(y|x)]` needs samples from the model itself, unlike cross-entropy 🔄
- Baselines and variance: what a group mean buys over a learned value function 📉
- The "choices that break the gradient": std advantage normalization and per-sequence length normalization 🕵️
- Importance weighting + clipping — how off-policy data buys wall-clock speed 🏎️
- How much of the final accuracy is the prompt, and how much is the algorithm 📐

## 📝 Field notes

A running log — deliberately empty for now, it fills up as the runs land.

- **Rewards are binary:** total = answer reward; format reward is for logging only (no partial credit).
- **Stop correctly:** `</answer>` stops generation for the `r1_zero*` prompts only — never for `question_only`.
- **Variance is the enemy:** run 4+ seeds and always plot the spread, not just the mean 🌫️
- **Reward starts near zero** — that's expected; watch the *trend* over ~50 steps, not step 1.
- **The bar:** ≥25% final validation accuracy, averaged across random seeds.

## ✅ Definition of done

- `uv run pytest tests/test_grpo.py` green 🟢
- Prompting baselines evaluated and characterized on GSM8K 🔍
- A GRPO loop whose validation reward rises, with rollouts that visibly improve 🚀
- LR sweep + prompt ablation plots, with honest commentary on variance 📈
- Dr. GRPO / RFT / MaxRL and off-policy comparisons, plus a write-up of my own estimator 🧪

## 🏃 Getting it running

1. `uv sync --no-install-package flash-attn && uv sync` — `flash-attn` is picky, so install it in two passes 🔧
2. `uv run pytest tests/test_grpo.py` — red everywhere at first, which is exactly right 🩸
3. The rollout loop talks to a **vLLM** server via `cs336_alignment/vllm_utils.py` (weights sync between steps) 🤗
4. Start from the suggested GRPO hyperparameters in the handout (group size 8, lr 1e-5, 200 rollout steps).

> 💡 The model is **OLMo-2-0425-1B**: train on `data/gsm8k/train.jsonl`, validate on `data/gsm8k/test.jsonl`.

## 🧭 Repo layout

- `cs336_alignment/` — my implementation (empty right now, apart from the provided starters) ✍️
- `cs336_alignment/vllm_utils.py` — vLLM server + weight-sync helper 🤗
- `cs336_alignment/drgrpo_grader.py` — reward functions for the `r1_zero` / `question_only` prompts 🏅
- `cs336_alignment/prompts/` — prompt templates (`r1_zero`, `question_only`, the 3-shot GSM8K prompt, …)
- `tests/adapters.py` + `tests/test_grpo.py` — the required suites (the safety tests are optional)
- `data/gsm8k/` · `scripts/` · `pyproject.toml` · `test_and_make_submission.sh` · the two PDFs

---

Reward the reasoning, not the words. 🎯

