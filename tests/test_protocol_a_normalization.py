from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genespt.preprocessing import (  # noqa: E402
    PROTOCOL_A_POLICY,
    ZERO_TRAIN_LIBRARY_POLICY,
    normalize_st_protocol_a,
    validate_gene_splits,
)


class ProtocolANormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = np.asarray(
            [
                [4.0, 3.0, 6.0, 2.0, 5.0],
                [0.0, 5.0, 0.0, 7.0, 9.0],
                [8.0, 0.0, 2.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        self.train_idx = np.asarray([0, 2], dtype=np.int64)
        self.val_idx = np.asarray([1], dtype=np.int64)
        self.test_idx = np.asarray([3, 4], dtype=np.int64)

    def normalize(
        self,
        counts: np.ndarray | None = None,
        **overrides: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        arguments: dict[str, object] = {
            "inner_train_gene_idx": self.train_idx,
            "val_gene_idx": self.val_idx,
            "test_gene_idx": self.test_idx,
            "require_complete_coverage": True,
        }
        arguments.update(overrides)
        return normalize_st_protocol_a(
            self.counts if counts is None else counts,
            **arguments,
        )

    def test_val_and_test_count_changes_do_not_affect_train_columns(self) -> None:
        changed = self.counts.copy()
        changed[:, self.val_idx] += np.asarray([[1000.0], [2000.0], [3000.0]])
        changed[:, self.test_idx] += np.asarray(
            [[4000.0, 5000.0], [6000.0, 7000.0], [8000.0, 9000.0]]
        )

        before, before_audit = self.normalize()
        after, after_audit = self.normalize(changed)

        np.testing.assert_array_equal(before[:, self.train_idx], after[:, self.train_idx])
        self.assertEqual(
            before_audit["denominator_gene_index_sha256"],
            after_audit["denominator_gene_index_sha256"],
        )
        self.assertFalse(np.array_equal(before[[0, 2], 1:], after[[0, 2], 1:]))

    def test_train_library_denominator_is_applied_to_every_column(self) -> None:
        normalized, _ = self.normalize()
        expected = np.zeros_like(self.counts, dtype=np.float64)
        expected[[0, 2]] = np.log1p(
            self.counts[[0, 2]].astype(np.float64) / 10.0 * 10_000.0
        )

        np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=0.0)

    def test_zero_train_library_spot_is_zeroed_and_audited(self) -> None:
        normalized, audit = self.normalize()
        expected_hash = hashlib.sha256(b"[0,2]").hexdigest()

        np.testing.assert_array_equal(
            normalized[1], np.zeros(self.counts.shape[1], dtype=np.float64)
        )
        self.assertEqual(audit["protocol"], "A")
        self.assertEqual(audit["policy"], PROTOCOL_A_POLICY)
        self.assertEqual(audit["shape"], [3, 5])
        self.assertEqual(audit["target_sum"], 10_000.0)
        self.assertEqual(audit["denominator_gene_count"], 2)
        self.assertEqual(audit["denominator_gene_index_sha256"], expected_hash)
        self.assertEqual(audit["zero_train_library_rows"], [1])
        self.assertEqual(audit["zero_train_library_spot_count"], 1)
        self.assertEqual(
            audit["zero_train_library_policy"], ZERO_TRAIN_LIBRARY_POLICY
        )
        self.assertEqual(json.loads(json.dumps(audit, sort_keys=True)), audit)

    def test_inner_train_indices_are_required_with_no_all_gene_fallback(self) -> None:
        with self.assertRaisesRegex(TypeError, "inner_train_gene_idx"):
            normalize_st_protocol_a(
                self.counts,
                val_gene_idx=self.val_idx,
                test_gene_idx=self.test_idx,
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.normalize(inner_train_gene_idx=None)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.normalize(inner_train_gene_idx=[])

    def test_illegal_split_types_ranges_shapes_and_overlaps_fail(self) -> None:
        with self.assertRaisesRegex(TypeError, "integers"):
            self.normalize(inner_train_gene_idx=np.asarray([0.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.normalize(val_gene_idx=np.asarray([-1], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.normalize(test_gene_idx=np.asarray([3, 5], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.normalize(inner_train_gene_idx=np.asarray([[0, 2]], dtype=np.int64))

        overlap_cases = (
            {
                "inner_train_gene_idx": [0, 2],
                "val_gene_idx": [2],
                "test_gene_idx": [3, 4],
            },
            {
                "inner_train_gene_idx": [0, 2],
                "val_gene_idx": [1],
                "test_gene_idx": [2, 4],
            },
            {
                "inner_train_gene_idx": [0, 2],
                "val_gene_idx": [1, 3],
                "test_gene_idx": [3, 4],
            },
        )
        for arguments in overlap_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.normalize(**arguments)

    def test_duplicate_indices_fail_in_every_split(self) -> None:
        duplicate_cases = (
            {"inner_train_gene_idx": [0, 0]},
            {"val_gene_idx": [1, 1]},
            {"test_gene_idx": [3, 3]},
        )
        for arguments in duplicate_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    self.normalize(**arguments)

    def test_complete_coverage_is_optional_and_enforced_when_requested(self) -> None:
        train, validation, test = validate_gene_splits(
            5,
            train_gene_idx=[0, 2],
            val_gene_idx=[1],
            test_gene_idx=[3, 4],
            require_complete_coverage=True,
        )
        np.testing.assert_array_equal(train, np.asarray([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(validation, np.asarray([1], dtype=np.int64))
        np.testing.assert_array_equal(test, np.asarray([3, 4], dtype=np.int64))

        validate_gene_splits(
            5,
            train_gene_idx=[0, 2],
            val_gene_idx=[1],
            test_gene_idx=[3],
            require_complete_coverage=False,
        )
        with self.assertRaisesRegex(ValueError, "completely cover"):
            validate_gene_splits(
                5,
                train_gene_idx=[0, 2],
                val_gene_idx=[1],
                test_gene_idx=[3],
                require_complete_coverage=True,
            )

    def test_normalization_defaults_to_complete_fold_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "completely cover"):
            normalize_st_protocol_a(
                self.counts,
                inner_train_gene_idx=[0, 2],
                val_gene_idx=[1],
                test_gene_idx=[3],
            )

    def test_nonfinite_and_negative_inputs_fail_closed(self) -> None:
        invalid_cases = (
            (np.nan, "finite"),
            (np.inf, "finite"),
            (-1.0, "nonnegative"),
        )
        for value, message in invalid_cases:
            changed = self.counts.copy()
            changed[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    self.normalize(changed)

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            self.normalize(self.counts[0])

    def test_target_sum_is_fixed_by_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires target_sum"):
            self.normalize(target_sum=5_000.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.normalize(target_sum=np.nan)

    def test_output_and_audit_are_deterministic_without_mutating_input(self) -> None:
        original = self.counts.copy()
        first, first_audit = self.normalize()
        second, second_audit = self.normalize()

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(self.counts, original)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(
            json.dumps(first_audit, sort_keys=True, separators=(",", ":")),
            json.dumps(second_audit, sort_keys=True, separators=(",", ":")),
        )


if __name__ == "__main__":
    unittest.main()
