"""Data plane: permutation, packing, and batch determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microlab.data.loader import DeterministicBatcher, SequentialBatcher
from microlab.data.packed import PackedDataset, ShardManifest, write_shard
from microlab.data.permutation import FeistelPermutation


class TestFeistelPermutation:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 17, 100, 1000, 4096, 5003])
    def test_is_a_bijection(self, n):
        """Every index maps to a distinct index in range — the whole point."""
        perm = FeistelPermutation(n, seed=42)
        out = perm(np.arange(n))
        assert out.min() >= 0 and out.max() < n
        assert len(np.unique(out)) == n

    def test_different_seeds_give_different_orders(self):
        n = 500
        a = FeistelPermutation(n, seed=1)(np.arange(n))
        b = FeistelPermutation(n, seed=2)(np.arange(n))
        assert not np.array_equal(a, b)
        # Both are still bijections.
        assert len(np.unique(a)) == n and len(np.unique(b)) == n

    def test_deterministic_across_instances(self):
        a = FeistelPermutation(1000, seed=7)(np.arange(1000))
        b = FeistelPermutation(1000, seed=7)(np.arange(1000))
        np.testing.assert_array_equal(a, b)

    def test_shuffles_rather_than_shifts(self):
        """A permutation that maps i -> i + c would pass a bijection test."""
        n = 2000
        out = FeistelPermutation(n, seed=3)(np.arange(n))
        assert (out == np.arange(n)).sum() < n * 0.01
        # Neighbouring indices should not stay neighbours.
        assert np.abs(np.diff(out)).mean() > n * 0.1

    def test_partial_query_matches_full(self):
        """Permuting a subset must agree with permuting everything and slicing."""
        perm = FeistelPermutation(777, seed=11)
        full = perm(np.arange(777))
        subset = perm(np.array([5, 100, 776]))
        np.testing.assert_array_equal(subset, full[[5, 100, 776]])

    def test_rejects_out_of_domain(self):
        perm = FeistelPermutation(10, seed=0)
        with pytest.raises(ValueError, match="out of domain"):
            perm(np.array([10]))
        with pytest.raises(ValueError, match="out of domain"):
            perm(np.array([-1]))

    def test_rejects_empty_domain(self):
        with pytest.raises(ValueError, match="positive"):
            FeistelPermutation(0, seed=0)


@pytest.fixture
def shard(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "toy.bin"
    write_shard(path, tokens, ShardManifest(n_tokens=len(tokens), source="test"))
    return path


class TestPackedDataset:
    def test_sample_windows_are_disjoint_and_shifted(self, shard):
        ds = PackedDataset(shard, seq_len=10)
        assert len(ds) == (1000 - 1) // 10
        x, y = ds.gather(np.array([0, 1]))
        np.testing.assert_array_equal(x[0], np.arange(0, 10))
        np.testing.assert_array_equal(y[0], np.arange(1, 11))
        np.testing.assert_array_equal(x[1], np.arange(10, 20))

    def test_targets_are_inputs_shifted_by_one(self, shard):
        ds = PackedDataset(shard, seq_len=16)
        x, y = ds.gather(np.array([3, 7]))
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

    def test_manifest_round_trips(self, tmp_path):
        path = tmp_path / "m.bin"
        write_shard(
            path,
            np.arange(500, dtype=np.uint16),
            ShardManifest(n_tokens=0, tokenizer_sha="abc123", source="unit-test", eos_id=9),
        )
        ds = PackedDataset(path, seq_len=8)
        assert ds.manifest is not None
        assert ds.manifest.tokenizer_sha == "abc123"
        # n_tokens is filled in by write_shard from the actual array.
        assert ds.manifest.n_tokens == 500

    def test_rejects_out_of_range_vocab(self, tmp_path):
        with pytest.raises(ValueError, match="uint16"):
            write_shard(
                tmp_path / "big.bin",
                np.array([70000], dtype=np.int64),
                ShardManifest(n_tokens=1),
            )

    def test_missing_file_message_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="prepare"):
            PackedDataset(tmp_path / "nope.bin", seq_len=8)

    def test_too_short_shard_rejected(self, tmp_path):
        path = tmp_path / "short.bin"
        write_shard(path, np.arange(4, dtype=np.uint16), ShardManifest(n_tokens=4))
        with pytest.raises(ValueError, match="too few"):
            PackedDataset(path, seq_len=64)


class TestDeterministicBatcher:
    def _batcher(self, shard, batch_size=4, seq_len=10, accum=1, seed=0):
        return DeterministicBatcher(
            PackedDataset(shard, seq_len), batch_size=batch_size, seed=seed, grad_accum_steps=accum
        )

    def test_batch_is_a_pure_function_of_step(self, shard):
        """The property the whole resume story rests on."""
        a = self._batcher(shard)
        b = self._batcher(shard)
        for step in (0, 1, 17, 999):
            np.testing.assert_array_equal(a.sample_indices(step), b.sample_indices(step))

    def test_independent_of_access_order(self, shard):
        """Asking for step 100 first must not change what step 5 returns.

        A stateful loader fails this; it is the mechanism by which a resumed run
        silently sees different data.
        """
        a = self._batcher(shard)
        first = a.sample_indices(5)
        a.sample_indices(100)
        a.sample_indices(3)
        np.testing.assert_array_equal(a.sample_indices(5), first)

    def test_epoch_covers_every_sample_exactly_once(self, shard):
        batch_size = 3
        b = self._batcher(shard, batch_size=batch_size)
        n = b.dataset.n_samples
        steps = n // batch_size
        seen = np.concatenate([b.sample_indices(s) for s in range(steps)])
        assert len(np.unique(seen)) == len(seen)  # no repeats within an epoch
        assert len(seen) == steps * batch_size

    def test_consecutive_epochs_differ(self, shard):
        # batch_size must divide n_samples (99) for epoch boundaries to fall on
        # step boundaries; otherwise an "epoch" of steps straddles the seam and
        # legitimately contains samples from both permutations.
        b = self._batcher(shard, batch_size=3)
        n = b.dataset.n_samples
        assert n % 3 == 0
        steps_per_epoch = n // 3
        first = np.concatenate([b.sample_indices(s) for s in range(steps_per_epoch)])
        second = np.concatenate(
            [b.sample_indices(s) for s in range(steps_per_epoch, 2 * steps_per_epoch)]
        )
        assert len(first) == n and len(second) == n
        assert sorted(first) == sorted(second)  # same samples
        assert not np.array_equal(first, second)  # different order

    def test_different_seeds_give_different_order(self, shard):
        a = self._batcher(shard, seed=1).sample_indices(0)
        b = self._batcher(shard, seed=2).sample_indices(0)
        assert not np.array_equal(a, b)

    def test_grad_accum_preserves_the_data_stream(self, shard):
        """Changing grad_accum at fixed global batch must see identical data.

        Otherwise a memory workaround silently becomes a different experiment,
        and two runs that should be comparable are not.
        """
        big = self._batcher(shard, batch_size=8, accum=1)
        small = self._batcher(shard, batch_size=4, accum=2)
        for step in (0, 1, 5):
            expected = big.sample_indices(step)
            got = np.concatenate([small.sample_indices(step, m) for m in range(2)])
            np.testing.assert_array_equal(got, expected)

    def test_batch_returns_correct_shapes_and_dtype(self, shard):
        b = self._batcher(shard, batch_size=4, seq_len=10)
        x, y = b.batch(0)
        assert x.shape == (4, 10) and y.shape == (4, 10)
        assert x.dtype == torch.long and y.dtype == torch.long

    def test_batch_contents_match_indices(self, shard):
        b = self._batcher(shard, batch_size=4, seq_len=10)
        idx = b.sample_indices(3)
        x, _ = b.batch(3)
        for row, i in enumerate(idx):
            assert int(x[row, 0]) == int(i) * 10

    def test_tokens_per_step_accounting(self, shard):
        b = self._batcher(shard, batch_size=4, seq_len=10, accum=3)
        assert b.tokens_per_optimizer_step == 4 * 3 * 10

    def test_negative_step_rejected(self, shard):
        with pytest.raises(ValueError, match="non-negative"):
            self._batcher(shard).sample_indices(-1)


class TestSequentialBatcher:
    def test_eval_batches_are_stable(self, shard):
        s = SequentialBatcher(PackedDataset(shard, 10), batch_size=4)
        x1, _ = s.batch(0)
        x2, _ = s.batch(0)
        torch.testing.assert_close(x1, x2)
        assert int(x1[0, 0]) == 0

    def test_wraps_around(self, shard):
        s = SequentialBatcher(PackedDataset(shard, 10), batch_size=4)
        torch.testing.assert_close(s.batch(0)[0], s.batch(s.n_batches)[0])

    def test_raises_when_val_shard_smaller_than_a_batch(self, tmp_path):
        path = tmp_path / "tiny.bin"
        write_shard(path, np.arange(50, dtype=np.uint16), ShardManifest(n_tokens=50))
        s = SequentialBatcher(PackedDataset(path, 10), batch_size=64)
        with pytest.raises(ValueError, match="fewer than one batch"):
            s.batch(0)
