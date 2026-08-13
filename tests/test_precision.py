"""Precision strategies. The fp16 path is the one that runs on Kaggle."""

from __future__ import annotations

import pytest
import torch

from microlab.precision import (
    BFloat16,
    Float16,
    FullPrecision,
    PrecisionStrategy,
    bf16_natively_supported,
    build_precision,
)


def _model_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 1))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


class TestFactory:
    def test_fp32_on_cpu(self):
        assert isinstance(build_precision("fp32", "cpu"), FullPrecision)

    def test_bf16_allowed_on_cpu(self):
        assert isinstance(build_precision("bf16", "cpu"), BFloat16)

    def test_fp16_on_cpu_is_rejected_with_guidance(self):
        """CPU fp16 autocast exists but has different numerics; silently
        accepting it would make CI unrepresentative of the T4 run."""
        with pytest.raises(ValueError, match="requires CUDA"):
            build_precision("fp16", "cpu")

    def test_unknown_precision_rejected(self):
        with pytest.raises(ValueError, match="unknown precision"):
            build_precision("fp8", "cpu")

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_bf16_rejected_on_turing(self):
        """On a T4 this must fail at startup, not 40 minutes into a run.

        The skip condition is capability-based on purpose. Written as
        ``if torch.cuda.is_bf16_supported(): skip``, this test skipped itself on
        a Kaggle T4 — the one machine it was written for — because that flag
        counts software emulation.
        """
        if bf16_natively_supported():
            pytest.skip("this GPU has native bf16")
        with pytest.raises(RuntimeError, match="no native bf16"):
            build_precision("bf16", "cuda")


class TestInterface:
    @pytest.mark.parametrize("name", ["fp32", "bf16"])
    def test_strategies_implement_the_interface(self, name):
        strategy = build_precision(name, "cpu")
        assert isinstance(strategy, PrecisionStrategy)
        for method in ("autocast", "backward", "clip_grad_norm", "step", "state_dict"):
            assert callable(getattr(strategy, method))

    @pytest.mark.parametrize("name", ["fp32", "bf16"])
    def test_full_step_runs(self, name):
        strategy = build_precision(name, "cpu")
        model, optimizer = _model_and_optimizer()
        with strategy.autocast():
            loss = model(torch.randn(4, 8)).square().mean()
        strategy.backward(loss)
        norm = strategy.clip_grad_norm(model, optimizer, 1.0)
        assert torch.isfinite(norm)
        assert strategy.step(optimizer) is True

    def test_fp32_state_dict_is_empty(self):
        assert build_precision("fp32", "cpu").state_dict() == {}

    def test_bf16_autocast_changes_dtype(self):
        strategy = build_precision("bf16", "cpu")
        model = torch.nn.Linear(8, 8)
        with strategy.autocast():
            out = model(torch.randn(2, 8))
        assert out.dtype == torch.bfloat16

    def test_clipping_actually_clips(self):
        strategy = build_precision("fp32", "cpu")
        model, optimizer = _model_and_optimizer()
        loss = model(torch.randn(4, 8) * 1000).square().mean()
        strategy.backward(loss)
        pre_clip = strategy.clip_grad_norm(model, optimizer, 1.0)
        assert pre_clip > 1.0
        post = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert post <= 1.0 + 1e-4


class TestFloat16State:
    """GradScaler state must round-trip: the loss scale is real training state.

    Restarting a resumed run at the default scale re-runs the scaler's backoff
    search, skipping a handful of steps after every single preemption. Over ~50
    sessions that is a few hundred wasted steps and a visible sawtooth in the
    scale plot.
    """

    def _strategy(self) -> Float16:
        # Constructed directly rather than through the factory so the state
        # logic is testable on CPU; only autocast/backward need CUDA.
        return Float16.__new__(Float16)

    def test_state_dict_round_trips(self):
        s = self._strategy()
        s.scaler = torch.amp.GradScaler("cpu", enabled=False)
        s.skipped_steps = 7
        s.total_steps = 100
        state = s.state_dict()

        restored = self._strategy()
        restored.scaler = torch.amp.GradScaler("cpu", enabled=False)
        restored.skipped_steps = 0
        restored.total_steps = 0
        restored.load_state_dict(state)

        assert restored.skipped_steps == 7
        assert restored.total_steps == 100

    def test_metrics_expose_skip_rate(self):
        s = self._strategy()
        s.scaler = torch.amp.GradScaler("cpu", enabled=False)
        s.skipped_steps = 5
        s.total_steps = 50
        metrics = s.metrics()
        assert metrics["skip_rate"] == pytest.approx(0.1)
        assert "loss_scale" in metrics

    def test_skip_rate_safe_at_zero_steps(self):
        s = self._strategy()
        s.scaler = torch.amp.GradScaler("cpu", enabled=False)
        s.skipped_steps = 0
        s.total_steps = 0
        assert s.metrics()["skip_rate"] == 0.0

    def test_load_state_dict_tolerates_old_checkpoints(self):
        """Checkpoints written before the counters existed must still load."""
        s = self._strategy()
        s.scaler = torch.amp.GradScaler("cpu", enabled=False)
        s.skipped_steps = 3
        s.total_steps = 3
        s.load_state_dict({"scaler": s.scaler.state_dict()})
        assert s.skipped_steps == 0


class TestNativeBF16Detection:
    """bf16 support must be decided by compute capability, not by PyTorch's flag.

    A real Kaggle T4 running torch 2.10 reports::

        gpu0: Tesla T4 sm_75 15.6GB bf16=True

    because `torch.cuda.is_bf16_supported()` counts software emulation. The
    guard that exists to stop someone selecting bf16 on Turing was built on
    that flag, so on the exact hardware this project runs on, it did not fire.
    These tests pin the capability-based check without needing a GPU.
    """

    @pytest.mark.parametrize(
        "capability,expected",
        [
            ((7, 5), False),  # Turing (T4) — emulated only
            ((6, 0), False),  # Pascal (P100)
            ((7, 0), False),  # Volta (V100)
            ((8, 0), True),   # Ampere (A100)
            ((8, 6), True),   # Ampere (A10G/3090)
            ((8, 9), True),   # Ada (L4/4090)
            ((9, 0), True),   # Hopper (H100)
        ],
    )
    def test_capability_decides(self, monkeypatch, capability, expected):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: capability)
        assert bf16_natively_supported() is expected

    def test_false_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert bf16_natively_supported() is False

    def test_turing_rejected_even_when_torch_claims_support(self, monkeypatch):
        """The exact Kaggle situation: emulation reported as support."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: (7, 5))
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=0: "Tesla T4")

        with pytest.raises(RuntimeError, match="no native bf16") as excinfo:
            BFloat16("cuda")
        # The message must explain the contradiction the user can see on screen.
        assert "emulated" in str(excinfo.value)

    def test_ampere_accepted(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: (8, 0))
        assert isinstance(BFloat16("cuda"), BFloat16)


class TestUnscaleOnceOnCPU:
    """The unscale-exactly-once property, checked without a GPU.

    This is the single most consequential invariant in the fp16 path: unscaling
    twice divides every gradient by the loss scale a second time, and the run
    trains to a visibly worse loss with nothing in the logs to explain it.
    Guarding it only behind an @gpu marker means it is checked whenever someone
    happens to run the suite on Kaggle — which is not a schedule.

    ``GradScaler`` supports CPU, so the scale/unscale bookkeeping (which is
    dtype-independent) is testable on every commit. What is *not* reproducible
    on CPU is fp16 overflow behaviour; that stays in the GPU tests.
    """

    def _strategy(self) -> Float16:
        strategy = Float16.__new__(Float16)
        strategy.device_type = "cpu"
        strategy.scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0**8)
        strategy.skipped_steps = 0
        strategy.total_steps = 0
        return strategy

    def test_clip_then_step_unscales_exactly_once(self):
        strategy = self._strategy()
        model = torch.nn.Linear(4, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        strategy.backward(model(torch.ones(1, 4)).sum())
        strategy.clip_grad_norm(model, optimizer, 1e9)  # effectively no clipping
        # d(sum(Wx))/dW with x == 1 is exactly 1 per element, once unscaled.
        torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight.grad))

        # step() must not unscale again. A second division would leave the
        # gradients at 1/scale of their true value.
        assert strategy.step(optimizer) is True
        torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight.grad))

    def test_step_without_clipping_also_unscales(self):
        """Skipping clip_grad_norm must not leave gradients scaled."""
        strategy = self._strategy()
        model = torch.nn.Linear(4, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        strategy.backward(model(torch.ones(1, 4)).sum())
        assert strategy.step(optimizer) is True
        # SGD with lr=1 subtracts the (unscaled) gradient exactly once.
        torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight.grad))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestFloat16OnGPU:
    def test_step_skipped_on_nonfinite_gradient(self):
        strategy = build_precision("fp16", "cuda")
        model = torch.nn.Linear(8, 8).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with strategy.autocast():
            loss = model(torch.randn(4, 8, device="cuda")).square().mean()
        strategy.backward(loss)
        for p in model.parameters():
            p.grad.fill_(float("inf"))
        strategy.clip_grad_norm(model, optimizer, 1.0)
        assert strategy.step(optimizer) is False
        assert strategy.skipped_steps == 1

    def test_gradients_are_unscaled_exactly_once(self):
        """Unscaling twice would shrink every gradient by the loss scale.

        The run still trains, just far worse, with nothing in the logs. This
        test pins the interface that prevents it.

        The scale is set well below the default here, and that matters: fp16
        tops out at 65504, so the default init_scale of 2**16 = 65536 is
        *already* inf in fp16. With dL/dW == 1, the scaled gradient overflows
        before unscaling happens at all, and the assertion below sees nan
        rather than the property it is trying to measure. Production keeps the
        high default on purpose — GradScaler halves it until it fits, which is
        exactly what test_step_skipped_on_nonfinite_gradient covers.
        """
        strategy = Float16("cuda", init_scale=2.0**8)
        model = torch.nn.Linear(4, 1, bias=False).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        x = torch.ones(1, 4, device="cuda")

        with strategy.autocast():
            loss = (model(x)).sum()
        strategy.backward(loss)
        strategy.clip_grad_norm(model, optimizer, 1e9)  # effectively no clipping
        # d(sum(Wx))/dW with x == 1 is exactly 1 per element, once unscaled.
        torch.testing.assert_close(
            model.weight.grad, torch.ones_like(model.weight.grad), rtol=1e-3, atol=1e-3
        )
