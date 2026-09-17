# 📈 CS336 Assignment 3 — Scaling: Fitting a Scaling Law

You're in charge of ClosedAI's next model, with a fixed compute budget and an Antarctic ice shelf on the
line. 🧊 The job: spend a **12 B200-hour** experiment budget on a training API, fit a scaling law relating
loss to model size and compute, and use it to predict the compute-optimal configuration (and final loss) for
the big **48 B200-hour** run. Handout: [`cs336_assignment3_scaling.pdf`](./cs336_assignment3_scaling.pdf).

**Progress** — ⬜⬜ **0 / 2 pieces** written (55 pts). Scaffolding only. ☕

## 📋 Build plan

- [ ] **IsoFLOPs scaling laws** · 5 pts — reproduce the IsoFLOPs analysis and derive the loss-optimal model
  size for a compute budget 📐
- [ ] **Scaling-law leaderboard** · 50 pts — design the ≤12 B200-hr query schedule, fit the law, then submit
  the predicted optimal hyperparameters + final validation loss 🔮

## 🧐 What I want to understand

- Kaplan vs. Hoffmann (Chinchilla): two parameterizations of the same size/data trade-off 📉
- IsoFLOPs profiles — how the loss-optimal model size slides as compute grows
- Non-embedding parameter counting: `12 · n_layer · d_model²`
- Budget allocation as the real problem: which runs are worth their API time? 💸
- How far an extrapolation beyond the fitted data can be trusted 🎯

## 📝 Field notes

A running log — deliberately empty for now, it fills up as runs land.

- **Hard cap:** the API enforces the 12 B200-hour fitting budget; the final run is allowed 48 B200-hours.
- **Resubmissible:** the `/final_submission` endpoint can be updated — the latest submission replaces the old.
- **Two submissions:** `writeup.pdf` + `code.zip` to Gradescope, *and* the predicted config/loss to the API.
- **Graded partly on outcome:** the predicted optimal model's real performance counts, not just the report.

## ✅ Definition of done

- A methodology write-up detailed enough to reproduce the fits 🔬
- A scaling law that fits the queried data well (with residuals I'm honest about) 📊
- Predicted optimal model size + hyperparameters for the 48 B200-hr budget 🔮
- Predicted final validation loss submitted to the API, alongside the writeup 📬

## 🏃 Getting it running

1. `uv sync` — deps are managed with [`uv`](https://docs.astral.sh/uv/).
2. `export A3_API_KEY=<your 8-digit student ID>` 🔑
3. Hit the hosted API at `http://hyperturing.stanford.edu:8000` — see the
   [docs](http://hyperturing.stanford.edu:8000/docs) and
   [dashboard](http://hyperturing.stanford.edu:8000/dashboard).
4. Start from [`examples/client_example.ipynb`](./examples/client_example.ipynb) to submit and inspect runs 🔁

> 💡 Offline / non-student? `uv sync --extra server`, pull tokenized data with
> `uv run modal run scripts/1_download_tokenized_data.py`, then `uv run cs336_scaling/training/run.py`.

## 🧭 Repo layout

- `cs336_scaling/client.py` — the API client (`get_budget`, `submit_experiment`, `list_experiments`, …)
- `cs336_scaling/schemas/` · `training/` · `scheduler/` · `api/` — request/response types and the training side
- `examples/client_example.ipynb` — minimal submit-and-inspect example
- `scripts/` · `tests/` — utilities and the scheduler/API tests
- `pyproject.toml` · `cs336_assignment3_scaling.pdf`

---

Extrapolate responsibly. 🔮
