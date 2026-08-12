# Compute

The whole project runs on free tiers, with a hard ceiling of $0–50 out of pocket.
Every estimate below is in **Kaggle-weeks**: 30 GPU-hours/week on 2xT4 at roughly
40 TFLOP/s effective, ≈ 4x10^18 FLOPs/week.

| Source | What you get | Used for |
| --- | --- | --- |
| **Kaggle** | 30 hrs/week GPU (2xT4, 32 GB combined), 20 hrs/week TPU, 12-hr sessions, uncapped CPU | Primary cluster — everything |
| **Modal** | $30/month free credit | The one multi-GPU benchmark Kaggle cannot give (M2) |
| **TRC** | Cloud TPU v3/v4, 30 days renewable, if accepted | The flagship pretrain (M4) |
| **HF Hub** | Free unlimited public storage | Shards and checkpoints — Kaggle's 20 GB is not enough |
| **GitHub Actions** | Free CI minutes | CPU correctness tests |

## The two constraints that shape the code

**T4 is Turing (sm_75).** fp16 only — no bf16, no FlashAttention-2. This is why
`precision` is a config-selected strategy rather than a hardcoded autocast, why
`BFloat16` refuses to construct on a Turing card at startup, and why attention
uses SDPA's memory-efficient backend explicitly instead of flash. See
`src/microlab/precision.py` and `src/microlab/model/attention.py`.

**Every session is preempted at 12 hours.** Rather than working around this, the
repo treats it as the normal case: wall-clock checkpoint cadence, commit markers
that make interrupted writes detectable, newest-first loading with fallback,
SIGTERM flush, and a run record that spans sessions. "Trained across ~50
preempted sessions with bit-exact resume" is a stronger engineering claim than
"trained on a rented H100 node" — but only if it is actually true, which is what
`tests/test_session_chaining.py` exists to keep honest.

## Budget by milestone

| | Milestone | Weeks | Kaggle-weeks of quota |
| --- | --- | --- | --- |
| M0 | End to end, tiny | 3 | 0.2 |
| M1 | Tokenizer + data | 12 | 4 |
| M2 | Systems | 12 | 2 (+$30 Modal) |
| M3 | Scaling laws | 6 | 2.5 |
| M4 | Flagship 500M | 9 | 7 (or ~4 TRC days) |
| M5 | Post-training | 7 | 1 |
| M6 | Inference + RL | 12 | 2 |

≈19 Kaggle-weeks of quota against ~61 calendar weeks available. **The binding
constraint is time, not compute** — so there is idle quota most weeks, and it
should go to extra seeds and ablations rather than being left on the table.

## Running on Kaggle

1. New Notebook → Settings → Accelerator: **GPU T4 x2**, Internet: **on**.
2. Clone and install:

   ```python
   !git clone https://github.com/achalshah20/microlab.git
   %cd microlab
   !pip install -q -e .
   ```

3. Build shards once and persist them (Kaggle's working directory does not
   survive, so push to HF Hub or write to a Kaggle Dataset):

   ```python
   !python -m microlab.cli prepare --source tinystories --out-dir /kaggle/working/data/tinystories --vocab-size 8192
   ```

4. Train. The run resumes automatically if a checkpoint exists, so re-running the
   same cell after a preemption continues rather than restarting:

   ```python
   !python -m microlab.cli train m0 data.train_bin=/kaggle/working/data/tinystories/train.bin
   ```

5. Verify the GPU-only tests on the real hardware:

   ```python
   !pytest -m gpu -q
   ```

`scripts/kaggle_train.py` wraps steps 3–4 with the session bookkeeping.

## Applications to send early

**TRC (TPU Research Cloud)** — apply *before* starting M3. Acceptance is not
instant, and M4 is 7 Kaggle-weeks on T4s versus ~4 days on a v3-8, so the
application timing is worth more than any optimization in M2. Ask for v3-8,
30 days renewable, and describe the project as an open, reproducible
from-scratch training run with published artifacts.

**Modal** — the $30/month credit is the only way this project gets a multi-GPU
data point. Spend it in M2 on a single scaling comparison, not on routine work
that Kaggle does for free.
