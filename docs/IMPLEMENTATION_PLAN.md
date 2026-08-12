# microlab — Implementation task list

> **Status (M0 implemented).** Phases 0–6 are built and tested; Phase 7's gate
> run is blocked on a GPU session. Deviations from the plan as written, all
> deliberate:
>
> - **Hydra + OmegaConf structured configs** replaced the plain-dataclass plan.
> - **fp16 unscaling takes the real optimizer**, not a proxy — the original
>   sketch would have made `GradScaler` unscale twice, silently shrinking every
>   gradient by the loss scale.
> - **`flops_per_token` gained an explicit LM-head term.** It omitted the tied
>   head and would have inflated every reported MFU by 16%; `tests/test_shapes.py`
>   now pins it against a per-module hook walk.
> - **The write-up is generated, not written** (`microlab shapes` →
>   `docs/M0_SHAPES.md`), so it cannot drift from the model.
> - **A synthetic corpus was added** so the whole pipeline runs offline and in
>   CI. It does not substitute for the TinyStories gate.


Derived from the roadmap doc. Scope of this document: **M0 in full detail**, plus the
cross-cutting foundations M0 forces us to design correctly, plus a thin note on how
M1–M7 attach. M1+ tasks are deliberately *not* expanded — the roadmap's stated failure
mode is breadth-first drift, and expanding them now is exactly that.

**M0 goal:** a complete training loop, from text file to sampled output, on one GPU.
**M0 gate:** 10M params on TinyStories → val loss < 1.6, grammatical multi-sentence
samples; resume matches an uninterrupted run to float tolerance; a deliberately killed
session recovers via the chaining harness.
**Budget:** ~6 GPU-hours (0.2 Kaggle-weeks), 3 calendar weeks, ~40 person-hours.

---

## Phase 0 — Repo skeleton and dev loop (~4 hrs)

- [ ] **0.1** `pyproject.toml` — package `microlab`, src layout, pinned `torch` (CUDA 12.x
      wheel that matches Kaggle's driver), `numpy`, `pyyaml`, `pytest`, `ruff`. Pin exactly;
      Kaggle's preinstalled torch drifts and will silently change numerics.
- [ ] **0.2** Directory skeleton:
      ```
      src/microlab/{cli,config,precision}.py
      src/microlab/model/{rmsnorm,rope,attention,mlp,transformer}.py
      src/microlab/data/{packed,loader,prepare_tinystories}.py
      src/microlab/train/{trainer,optim,checkpoint,chaining}.py
      src/microlab/tokenizer/minbpe.py
      src/microlab/eval/{sample,valloss}.py
      configs/m0.yaml   tests/   docs/
      ```
- [ ] **0.3** `ruff` + `pytest` config; `make test` / `make lint` (or a `scripts/` equivalent).
- [ ] **0.4** GitHub Actions: CPU-only job running the full test suite on push. Everything in
      M0 except the actual training run must be CPU-testable — that constraint is what makes
      correctness tests cheap for the next 14 months.
- [ ] **0.5** `README.md` stub + `docs/COMPUTE.md` stub (Kaggle/Modal/TRC/HF-Hub application
      notes, per the roadmap's compute model table).

## Phase 1 — Config, precision, determinism (~5 hrs)

These three are the load-bearing abstractions. Every later milestone depends on them and
retrofitting any of them is expensive.

- [ ] **1.1** `config.py` — nested dataclasses (`ModelCfg`, `DataCfg`, `OptimCfg`, `TrainCfg`,
      `RunCfg`), YAML load, CLI dotted-key overrides (`--optim.lr=3e-4`), unknown-key = hard
      error, and a `to_dict()` that goes verbatim into the checkpoint and run record.
- [ ] **1.2** `precision.py` — `PrecisionStrategy` selected by one config field
      (`precision: fp16 | bf16 | fp32`), exposing `autocast_ctx()`, `scale_loss()`,
      `unscale_and_clip_()`, `step()`, `state_dict()`/`load_state_dict()`.
      - fp16 path: `GradScaler`, **loss-scale logged every step** and an inf/nan-step counter
        surfaced as a first-class metric (T4 is Turing — fp16 only, no bf16, no FA2).
      - bf16 path: no scaler; same interface. Must be exercisable on CPU in tests.
      - Guard: refuse `bf16` at startup if `torch.cuda.is_bf16_supported()` is False.
- [ ] **1.3** Determinism utilities — `seed_everything(seed)`, per-rank/per-worker seed
      derivation, and a documented policy on which nondeterminism we accept (cuDNN algo
      selection) vs. forbid (data order, init, dropout).
- [ ] **1.4** Run record: `runs/<run_id>/` holding resolved config, git SHA + dirty flag,
      `metrics.jsonl`, and a session log. JSONL first, optional W&B behind a flag — no
      dependency on a hosted tracker.

## Phase 2 — Model (~8 hrs)

- [ ] **2.1** `RMSNorm` (pre-norm placement), fp32 accumulation inside the norm.
- [ ] **2.2** RoPE — precomputed cos/sin cache, correct dtype handling, applied to q/k only.
- [ ] **2.3** GQA causal attention on `torch.nn.functional.scaled_dot_product_attention`,
      memory-efficient backend explicitly selected (**not** FA2 — unavailable on Turing).
      KV-head repeat via `expand`, no materialized copy.
- [ ] **2.4** SwiGLU MLP with the standard `hidden = 8/3 * d_model` rounded to a multiple of 64.
- [ ] **2.5** `Transformer`: embedding, N blocks, final norm, LM head, optional weight tying;
      `init_weights()` with a documented scheme (std, residual-scaled output projections);
      `num_params()` and `flops_per_token()` helpers — the write-up needs them and so does
      M2's MFU math.
- [ ] **2.6** `configs/m0.yaml` — a ~10M-param TinyStories config (e.g. d_model 256,
      8 layers, 8 heads / 2 kv-heads, seq 512, vocab from the M0 tokenizer).

## Phase 3 — Tokenizer placeholder + data plane (~6 hrs)

- [ ] **3.1** `minbpe.py` — minimal byte-level BPE train/encode/decode, pure Python.
      Explicitly labeled a placeholder; the Rust implementation lands in M1.
      Round-trip test on unicode, emoji, and whitespace runs.
- [ ] **3.2** `prepare_tinystories.py` — download, train tokenizer (~8–16K vocab), encode to
      a flat `uint16` `.bin` + a small JSON manifest (token count, vocab hash, tokenizer hash,
      source revision). Train/val split written separately.
- [ ] **3.3** `packed.py` — `np.memmap` reader over the packed shard; no per-epoch shuffle
      buffer, just index math.
- [ ] **3.4** `loader.py` — **`(seed, step) → batch` is a pure function.** Batch indices derive
      from a counter-based RNG (e.g. `np.random.Philox`), not from iterator state. This is what
      makes resume bit-exact and it is the single most important design choice in M0.
- [ ] **3.5** Test: `batch(seed, step)` is stable across process restarts, worker counts, and
      batch-size-preserving reshuffles.

## Phase 4 — Trainer (~6 hrs)

- [ ] **4.1** AdamW with decay/no-decay param groups (no decay on norms and biases), fused
      impl when available.
- [ ] **4.2** Cosine schedule with linear warmup, computed from **global step**, so a resumed
      run lands on the identical LR.
- [ ] **4.3** Gradient accumulation with correct loss normalization; grad clipping applied
      after unscaling on the fp16 path.
- [ ] **4.4** Train step: forward, loss, backward, clip, step, zero-grad; per-step metrics —
      loss, LR, grad norm, tokens/sec, loss scale, step time, peak memory.
- [ ] **4.5** Periodic val loss + a sampling hook (temperature/top-k) that writes samples into
      the run record, so sample quality is visible during the run and not just at the gate.

## Phase 5 — Checkpointing and session chaining (~7 hrs)

The roadmap calls this the layer hobby repos skip and labs treat as table stakes. M4's
~50-session chain depends entirely on this working.

- [ ] **5.1** `checkpoint.py` — save model, optimizer, scheduler, scaler, global step,
      **CPU + CUDA RNG state**, data-loader position, and resolved config. Atomic write
      (tmp file → `os.replace`), keep last N + best.
- [ ] **5.2** Save cadence by **wall-clock minutes**, not steps — the constraint is a 12-hour
      preemption, not a step count.
- [ ] **5.3** `chaining.py` — a `run_id` that survives sessions; on start, detect an existing
      run and resume from the latest valid checkpoint; append to the same `metrics.jsonl` and
      session log rather than starting a new record; increment a session counter.
- [ ] **5.4** Corruption handling: verify the checkpoint (hash/complete-read) and fall back to
      the previous one on failure. A 50-session chain will hit a truncated write eventually.
- [ ] **5.5** `SIGTERM`/`SIGINT` handler that flushes a final checkpoint if time allows.
- [ ] **5.6** `microlab train configs/m0.yaml` + `microlab sample` CLI entry points.

## Phase 6 — Tests and correctness (~5 hrs)

The roadmap's stated M0 risk is silent correctness bugs. These come *before* trusting any
loss curve.

- [ ] **6.1** Causal mask test: token *t*'s logits are unchanged when tokens > *t* are perturbed.
- [ ] **6.2** RoPE vs. an independent reference implementation (complex-rotation form),
      including the position-offset case used at decode time.
- [ ] **6.3** GQA test: with `n_kv_heads == n_heads` the block matches plain MHA exactly.
- [ ] **6.4** **HF logit parity** — load identical weights into an equivalent
      `LlamaConfig` model and diff logits at fp32 tolerance. This is the test that catches the
      bugs the loss curve will hide.
- [ ] **6.5** Determinism: two runs, same seed, N steps → identical loss sequence.
- [ ] **6.6** Resume: run 2N steps uninterrupted vs. N + kill + resume-to-2N → losses match to
      float tolerance, weights match bitwise where the precision policy allows.
- [ ] **6.7** Precision-strategy tests on CPU for both paths; loss-scale backoff exercised with
      a synthetic inf.
- [ ] **6.8** Tokenizer round-trip and checkpoint round-trip tests.

## Phase 7 — Gate run and write-up (~5 hrs + ~6 GPU-hrs)

- [ ] **7.1** Kaggle notebook/script that clones the repo, installs, and launches training —
      the entry point for every later milestone's runs.
- [ ] **7.2** Full 10M-param TinyStories run to the gate. Target val loss **< 1.6**.
- [ ] **7.3** **Deliberately kill the session mid-run** and verify the chain recovers and the
      run record is continuous. Record this in the run log — it's the evidence, not a formality.
- [ ] **7.4** Sample dump: grammatical multi-sentence stories, published in the write-up.
- [ ] **7.5** Write-up: *"Every tensor shape in a forward pass"* — annotated shapes and FLOP
      counts per op, tied back to `flops_per_token()`.
- [ ] **7.6** README: what M0 is, how to reproduce, honest statement of what is placeholder
      (the tokenizer) and what is real.

---

## Explicitly NOT in M0

Listed so they get refused rather than drifted into: Rust tokenizer, WARC ingestion, dedup,
Triton kernels, FSDP/TP/PP, muP, scaling-law ladder, MoE, eval harness beyond val loss and
eyeballed samples. Multi-GPU is M2 — M0 is single-T4 and should stay that way, though nothing
in the code should *assume* world size 1.

## Decisions I've assumed (flag if you disagree)

1. **Config:** plain dataclasses + YAML, no Hydra/OmegaConf. Fewer moving parts, and the
   resolved config is trivially serializable into checkpoints.
2. **Tracking:** JSONL in the run directory as the source of truth; W&B optional behind a
   flag. Keeps the free-tier story clean and keeps runs reproducible offline.
3. **Placeholder tokenizer in Python**, not Rust — the roadmap puts Rust in M1, and an M0
   Rust build step would slow the Kaggle loop for no gain.
4. **Counter-based RNG for batch sampling** rather than a stateful sampler whose state gets
   checkpointed. Slightly unusual, but it makes bit-exact resume a property of the design
   instead of a thing to debug across 50 sessions.
5. **Torch pinned to a Kaggle-compatible CUDA wheel**, and CI runs CPU-only.

## Open questions for you

- **HF Hub account/org name** for shards and checkpoints (needed from M1, but worth reserving
  now so the config field has a real default).
- **TRC application timing** — the roadmap says apply before M3 starts. Want me to draft the
  application text as part of `docs/COMPUTE.md` in M0, so it's ready early?
- **Kaggle vs. local dev split** — is there any local GPU at all, or is Kaggle the only
  accelerator? It changes how aggressively Phase 6 needs to be CPU-runnable.

## Sequencing note

Phases 0→1→2 can be done in any order internally, but **Phase 1.2 (precision) and 3.4
(counter-based batching) must land before Phase 5**, or checkpoint/resume gets rewritten.
Phase 6.4 (HF logit parity) should run before the Phase 7 gate run, not after.
