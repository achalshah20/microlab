"""Shape and FLOP annotation for a forward pass.

This generates the M0 write-up deliverable ("every tensor shape in a forward
pass") by *instrumenting the model* rather than by describing it in prose. A
hand-written table is wrong the first time someone changes ``ffn_multiple_of``;
this one is regenerated from the model that actually runs.

FLOP convention matches ``Transformer.flops_per_token``: a multiply-accumulate
is 2 FLOPs, and the numbers here are forward-only. Elementwise work (norms,
SiLU, residual adds) is reported as ~0 because it is memory-bound, not
compute-bound — which is exactly why M2's kernel work targets it for *bandwidth*
rather than FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .config import ModelConfig
from .model.attention import CausalSelfAttention
from .model.transformer import Transformer


@dataclass
class ModuleRecord:
    name: str
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    params: int
    flops: float
    note: str = ""


@dataclass
class ShapeReport:
    config: ModelConfig
    batch_size: int
    seq_len: int
    records: list[ModuleRecord] = field(default_factory=list)
    total_params: int = 0
    non_embedding_params: int = 0
    analytic_flops_per_token: float = 0.0

    @property
    def measured_flops(self) -> float:
        return sum(r.flops for r in self.records)

    @property
    def measured_flops_per_token(self) -> float:
        return self.measured_flops / (self.batch_size * self.seq_len)


def _linear_flops(module: nn.Linear, out_shape: tuple[int, ...]) -> float:
    # 2 * (elements produced) * (input features contracted)
    produced = 1
    for d in out_shape:
        produced *= d
    return 2.0 * produced * module.in_features


def _attention_score_flops(module: CausalSelfAttention, b: int, s: int) -> float:
    """QK^T and AV, the two matmuls that are quadratic in sequence length.

    Note this counts the *full* S x S score matrix. Causal masking means only
    half of it is used, and a fused causal kernel (M2) skips the masked blocks
    entirely — so the achievable FLOP count is roughly half this. Reporting the
    full count is the standard convention and keeps MFU comparable with
    published numbers, but it is why a causal-aware kernel can appear to exceed
    100% of "expected" utilization.
    """
    return 2.0 * 2.0 * b * module.n_heads * s * s * module.head_dim


def profile_shapes(
    cfg: ModelConfig, batch_size: int = 1, seq_len: int | None = None
) -> ShapeReport:
    """Run one forward pass with hooks, recording every module's shapes and FLOPs."""
    seq_len = seq_len or cfg.max_seq_len
    model = Transformer(cfg).eval()
    report = ShapeReport(config=cfg, batch_size=batch_size, seq_len=seq_len)

    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(mod, inputs, output):
            in_t = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            out_t = output if torch.is_tensor(output) else None
            in_shape = tuple(in_t.shape) if in_t is not None else ()
            out_shape = tuple(out_t.shape) if out_t is not None else ()
            params = sum(p.numel() for p in mod.parameters(recurse=False))

            flops = 0.0
            note = ""
            if isinstance(mod, nn.Linear):
                flops = _linear_flops(mod, out_shape)
            elif isinstance(mod, nn.Embedding):
                note = "lookup, no FLOPs"
            elif isinstance(mod, CausalSelfAttention):
                flops = _attention_score_flops(mod, batch_size, seq_len)
                note = "QK^T + AV only; projections counted separately"
            else:
                note = "elementwise / memory-bound"

            report.records.append(
                ModuleRecord(
                    name=name,
                    kind=type(mod).__name__,
                    input_shape=in_shape,
                    output_shape=out_shape,
                    params=params,
                    flops=flops,
                    note=note,
                )
            )

        return hook

    for name, module in model.named_modules():
        if name == "" or isinstance(module, (nn.ModuleList, nn.Dropout)):
            continue
        handles.append(module.register_forward_hook(make_hook(name, module)))

    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        model(idx)
    for h in handles:
        h.remove()

    report.total_params = model.num_params(non_embedding=False)
    report.non_embedding_params = model.num_params(non_embedding=True)
    report.analytic_flops_per_token = model.flops_per_token(seq_len, backward=False)
    return report


def _fmt_shape(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]" if shape else "-"


def _fmt_num(n: float) -> str:
    for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return f"{n:.0f}"


def render_markdown(report: ShapeReport, first_block_only: bool = True) -> str:
    """Render the report as a Markdown document."""
    cfg = report.config
    lines: list[str] = []
    lines.append("# Every tensor shape in a forward pass\n")
    lines.append(
        "Generated by `microlab shapes` — regenerate rather than edit, so it "
        "cannot drift from the model.\n"
    )
    lines.append("## Configuration\n")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for key in (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "max_seq_len",
        "ffn_multiple_of",
        "tie_embeddings",
    ):
        lines.append(f"| {key} | {getattr(cfg, key)} |")
    head_dim = cfg.d_model // cfg.n_heads
    lines.append(f"| head_dim | {head_dim} |")
    lines.append(f"| n_rep (GQA) | {cfg.n_heads // cfg.n_kv_heads} |")
    lines.append(f"| batch x seq | {report.batch_size} x {report.seq_len} |")
    lines.append("")

    lines.append("## Parameters\n")
    lines.append(f"- total: **{report.total_params:,}** ({_fmt_num(report.total_params)})")
    lines.append(
        f"- non-embedding: **{report.non_embedding_params:,}** "
        f"({_fmt_num(report.non_embedding_params)}) — the count scaling-law fits use"
    )
    lines.append("")

    lines.append("## Forward pass\n")
    if first_block_only and cfg.n_layers > 1:
        lines.append(
            f"Blocks 1..{cfg.n_layers - 1} are identical to block 0 and are omitted; "
            "totals below cover all layers.\n"
        )
    lines.append("| module | type | input | output | params | FLOPs | note |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    seen: set[str] = set()
    for rec in report.records:
        if first_block_only:
            if rec.name.startswith("blocks.") and not rec.name.startswith("blocks.0"):
                continue
            # The RoPE cache is one shared module invoked once per block, so it
            # fires N times under the same name. Show it once.
            if rec.name in seen:
                continue
            seen.add(rec.name)
        lines.append(
            f"| `{rec.name or 'model'}` | {rec.kind} | {_fmt_shape(rec.input_shape)} | "
            f"{_fmt_shape(rec.output_shape)} | {rec.params:,} | "
            f"{_fmt_num(rec.flops) if rec.flops else '-'} | {rec.note} |"
        )
    lines.append("")

    lines.append("## FLOP accounting\n")
    measured = report.measured_flops_per_token
    analytic = report.analytic_flops_per_token
    lines.append(f"- measured (summed over modules): **{_fmt_num(measured)}** FLOPs/token forward")
    lines.append(f"- analytic (`flops_per_token`): **{_fmt_num(analytic)}** FLOPs/token forward")
    lines.append(f"- ratio measured/analytic: **{measured / analytic:.3f}**")
    lines.append("")
    lines.append(
        "The two should agree to within rounding. They are computed independently — "
        "the measured column sums per-module shapes captured by forward hooks, the "
        "analytic figure comes from `Transformer.flops_per_token`, which is what MFU "
        "is divided by. A gap between them means the MFU denominator is wrong: the "
        "first version of this table disagreed by 16%, which was `flops_per_token` "
        "omitting the tied LM head (it is excluded from the non-embedding parameter "
        "count, because under weight tying the head *is* the embedding matrix). "
        "Every reported MFU would have been inflated by that factor."
    )
    lines.append("")
    lines.append(
        f"Training FLOPs/token (forward + backward, 3x): **{_fmt_num(analytic * 3)}**"
    )
    return "\n".join(lines) + "\n"


def shape_report_dict(report: ShapeReport) -> dict[str, Any]:
    return {
        "batch_size": report.batch_size,
        "seq_len": report.seq_len,
        "total_params": report.total_params,
        "non_embedding_params": report.non_embedding_params,
        "measured_flops_per_token": report.measured_flops_per_token,
        "analytic_flops_per_token": report.analytic_flops_per_token,
        "modules": [
            {
                "name": r.name,
                "kind": r.kind,
                "input": list(r.input_shape),
                "output": list(r.output_shape),
                "params": r.params,
                "flops": r.flops,
            }
            for r in report.records
        ],
    }
