"""Deterministic batching: ``(seed, step) -> batch`` as a pure function.

This is the single most load-bearing design choice in M0. Every conventional
data loader keeps *state* — an iterator position, a shuffle buffer, a worker's
RNG — and resuming a run means serializing that state and hoping it round-trips.
Across ~50 preempted sessions (M4) that hope is misplaced: the failure mode is
not a crash but a run that quietly re-reads the same shard prefix after every
preemption, which looks like a slightly-too-good loss curve and invalidates the
whole run.

Here, the batch for a given step is computed from ``(seed, step)`` and nothing
else. There is no sampler state to checkpoint, no worker seeding to get wrong,
and resume correctness is a property of the design rather than a thing to debug.
The global step already lives in the checkpoint, so restoring the data stream is
free.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.determinism import derive_seed
from .packed import PackedDataset
from .permutation import FeistelPermutation


class DeterministicBatcher:
    """Maps a global step to a batch of samples, statelessly.

    Sample ordering is an epoch-wise permutation: global sample index
    ``g = step * batch_size + j`` splits into ``epoch = g // n_samples`` and a
    within-epoch offset, and the offset is mapped through a Feistel permutation
    keyed by ``(seed, epoch)``. Consequences:

    * Every sample is visited exactly once per epoch (no sampling-with-
      replacement coverage loss).
    * Each epoch has a different, unrelated order.
    * Nothing is stored: a batch is recomputed identically on any machine.
    """

    def __init__(
        self,
        dataset: PackedDataset,
        batch_size: int,
        seed: int,
        grad_accum_steps: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.grad_accum_steps = grad_accum_steps
        self._perm_cache: dict[int, FeistelPermutation] = {}

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.batch_size * self.grad_accum_steps * self.dataset.seq_len

    def _permutation(self, epoch: int) -> FeistelPermutation:
        if epoch not in self._perm_cache:
            # Bounded cache: a run only ever touches a handful of epochs, but a
            # long M4 run should not accumulate objects indefinitely.
            if len(self._perm_cache) > 8:
                self._perm_cache.clear()
            self._perm_cache[epoch] = FeistelPermutation(
                self.dataset.n_samples, derive_seed(self.seed, "epoch", epoch)
            )
        return self._perm_cache[epoch]

    def sample_indices(self, step: int, micro_step: int = 0) -> np.ndarray:
        """Dataset indices for one micro-batch. Pure function of the arguments."""
        if step < 0 or micro_step < 0:
            raise ValueError("step and micro_step must be non-negative")

        # Micro-batches within a gradient-accumulation group are consecutive
        # slices of the global stream, so changing grad_accum_steps at fixed
        # global batch size sees exactly the same data in the same order. That
        # is what makes an accumulation change a pure memory/throughput trade
        # and not a change to the experiment.
        flat_step = step * self.grad_accum_steps + micro_step
        base = flat_step * self.batch_size
        globals_ = np.arange(base, base + self.batch_size, dtype=np.int64)

        n = self.dataset.n_samples
        epochs = globals_ // n
        offsets = globals_ % n

        out = np.empty(self.batch_size, dtype=np.int64)
        # A batch straddles an epoch boundary at most once, so this loop runs
        # once in the common case and twice at the seam.
        for epoch in np.unique(epochs):
            mask = epochs == epoch
            out[mask] = self._permutation(int(epoch))(offsets[mask])
        return out

    def batch(
        self,
        step: int,
        micro_step: int = 0,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, y)`` int64 tensors of shape ``[batch_size, seq_len]``."""
        idx = self.sample_indices(step, micro_step)
        x_np, y_np = self.dataset.gather(idx)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if str(device) != "cpu":
            # pin+non_blocking overlaps the H2D copy with compute. Worth ~2% on
            # a T4, where the PCIe link is the narrowest part of the machine.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y

    def epoch_at_step(self, step: int) -> float:
        """Fractional epochs consumed after ``step`` optimizer steps."""
        seen = step * self.grad_accum_steps * self.batch_size
        return seen / self.dataset.n_samples


class SequentialBatcher:
    """Fixed, non-shuffled batches for evaluation.

    Validation must see the same samples in the same order at every eval, or the
    val curve moves for reasons that have nothing to do with the model. This
    walks the val shard from index 0 and is a pure function of the batch number.
    """

    def __init__(self, dataset: PackedDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = dataset.n_samples // batch_size

    def batch(
        self, index: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_batches == 0:
            raise ValueError(
                f"val shard has {self.dataset.n_samples} samples, "
                f"fewer than one batch of {self.batch_size}"
            )
        start = (index % self.n_batches) * self.batch_size
        idx = np.arange(start, start + self.batch_size, dtype=np.int64)
        x_np, y_np = self.dataset.gather(idx)
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
        if str(device) != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y
