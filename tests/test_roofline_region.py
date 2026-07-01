"""Tests for the roofline-region labeling logic."""

import numpy as np

from carm_paraver.GUI_utils import roofline_region_label


def test_roofline_region_label_all_three_regions() -> None:
    """Label AI values against L1 ridge=1.0 and DRAM ridge=10.0.

    Expected:
        1 (Memory Bound)  for AI < 1.0
        2 (Mixed)          for 1.0 <= AI <= 10.0
        3 (Compute Bound)  for AI > 10.0
    """
    ai = np.array([0.5, 1.0, 5.0, 10.0, 12.0])
    l1_ridge_x = 1.0
    dram_ridge_x = 10.0

    labels = roofline_region_label(ai, l1_ridge_x, dram_ridge_x)

    expected = np.array([1, 2, 2, 2, 3])
    np.testing.assert_array_equal(labels, expected)


def test_roofline_region_label_all_memory_bound() -> None:
    ai = np.array([0.0, 0.1, 0.99])
    labels = roofline_region_label(ai, l1_ridge_x=1.0, dram_ridge_x=10.0)
    np.testing.assert_array_equal(labels, [1, 1, 1])


def test_roofline_region_label_all_compute_bound() -> None:
    ai = np.array([10.1, 100.0, 1e6])
    labels = roofline_region_label(ai, l1_ridge_x=1.0, dram_ridge_x=10.0)
    np.testing.assert_array_equal(labels, [3, 3, 3])


def test_roofline_region_label_equal_ridges() -> None:
    """When l1_ridge_x == dram_ridge_x, only the exact point is Mixed."""
    ai = np.array([0.5, 5.0, 5.0, 10.0])
    labels = roofline_region_label(ai, l1_ridge_x=5.0, dram_ridge_x=5.0)
    expected = np.array([1, 2, 2, 3])
    np.testing.assert_array_equal(labels, expected)
