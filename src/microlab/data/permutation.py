"""Stateless pseudo-random permutations via a Feistel network.

The data loader needs to visit every training sample exactly once per epoch, in
an order that looks random, derived purely from ``(seed, epoch)`` with no state
to checkpoint and no array to materialize.

``np.random.permutation`` fails the last two requirements: at M4 scale (500M
params, 10B tokens, seq 2048) an epoch has ~5M samples, so the permutation array
is tens of MB that must be rebuilt identically on every one of ~50 resumed
sessions. Sampling with replacement instead (what nanoGPT does) avoids the state
but gives up exact coverage — at one epoch, ~37% of the corpus is never seen and
some samples are seen three times.

A Feistel network gives a keyed bijection on ``[0, 2^bits)`` computed in O(1)
per index with no memory. Cycle-walking (re-encrypting any output that lands
outside ``[0, n)``) narrows it to an exact permutation of the real domain. This
is the standard construction for format-preserving encryption; here the
"plaintext" is a sample index.
"""

from __future__ import annotations

import numpy as np

_SPLITMIX_A = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_B = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


class FeistelPermutation:
    """A deterministic bijection on ``[0, n)`` keyed by ``seed``.

    Args:
        n: domain size.
        seed: permutation key. Different seeds give unrelated orderings.
        rounds: Feistel rounds. Four is the standard minimum for a construction
            that looks random; this is a shuffle, not a cipher, so more rounds
            buy nothing observable.
    """

    def __init__(self, n: int, seed: int, rounds: int = 4) -> None:
        if n <= 0:
            raise ValueError(f"permutation domain must be positive, got {n}")
        self.n = n
        self.seed = seed
        self.rounds = rounds

        # Domain is rounded up to an even power of two so it splits into two
        # equal halves; cycle-walking maps it back down to exactly n.
        bits = max(2, int(np.ceil(np.log2(max(n, 2)))))
        if bits % 2:
            bits += 1
        self.bits = bits
        self.half_bits = bits // 2
        self.half_mask = np.uint64((1 << self.half_bits) - 1)

        self.round_keys = [
            np.uint64((seed * (r + 1) + 0x5DEECE66D) & 0xFFFFFFFFFFFFFFFF) for r in range(rounds)
        ]

    def _round_fn(self, x: np.ndarray, r: int) -> np.ndarray:
        """SplitMix64 finalizer keyed per round, truncated to half width."""
        z = (x + self.round_keys[r] + _GOLDEN).astype(np.uint64)
        z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_A
        z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_B
        z = z ^ (z >> np.uint64(31))
        return z & self.half_mask

    def _encrypt(self, x: np.ndarray) -> np.ndarray:
        left = (x >> np.uint64(self.half_bits)) & self.half_mask
        right = x & self.half_mask
        for r in range(self.rounds):
            left, right = right, left ^ self._round_fn(right, r)
        return (left << np.uint64(self.half_bits)) | right

    def __call__(self, indices: np.ndarray | int) -> np.ndarray:
        """Map indices in ``[0, n)`` to their permuted positions in ``[0, n)``."""
        signed = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        if np.any(signed < 0) or np.any(signed >= self.n):
            raise ValueError(f"index out of domain [0, {self.n})")

        # Wraparound is the intended semantics of every multiply and add in the
        # round function, so overflow warnings are noise here, not signal.
        with np.errstate(over="ignore"):
            out = self._encrypt(signed.astype(np.uint64))
            # Cycle-walk: encryption is a bijection on [0, 2^bits), so iterating
            # it from an out-of-range point traverses a cycle that must contain
            # an in-range point. The domain is < 2n, so this converges fast.
            for _ in range(64):
                out_of_range = out >= np.uint64(self.n)
                if not out_of_range.any():
                    break
                out[out_of_range] = self._encrypt(out[out_of_range])
            else:  # pragma: no cover - unreachable for any realistic n
                raise RuntimeError("Feistel cycle-walking failed to converge")
        return out.astype(np.int64)
