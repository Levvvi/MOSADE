"""Tests for decomposition: weight vectors, neighborhoods, scalarising."""

import numpy as np
import pytest

from mosade.algorithm.decomposition import (
    auto_partitions,
    compute_neighbors,
    das_dennis,
    tchebycheff,
    associate_to_weights,
)


class TestDasDennis:
    def test_2obj_returns_correct_count(self):
        # C(H + M - 1, M - 1) = C(9+1, 1) = 10
        w = das_dennis(9, 2)
        assert w.shape == (10, 2)

    def test_weights_sum_to_one(self):
        w = das_dennis(12, 3)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)

    def test_3obj_shape(self):
        w = das_dennis(5, 3)
        # C(5+2, 2) = 21
        assert w.shape == (21, 3)

    def test_all_non_negative(self):
        w = das_dennis(10, 4)
        assert np.all(w >= 0)


class TestNeighbors:
    def test_shape(self):
        w = das_dennis(9, 2)
        nb = compute_neighbors(w, T=3)
        assert nb.shape == (10, 3)

    def test_no_self_in_neighbors(self):
        w = das_dennis(9, 2)
        nb = compute_neighbors(w, T=3)
        for i in range(w.shape[0]):
            assert i not in nb[i]


class TestTchebycheff:
    def test_scalar(self):
        f = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        z = np.array([0.0, 0.0])
        # max(0.5*1, 0.5*2) = 1.0
        assert tchebycheff(f, w, z) == pytest.approx(1.0)

    def test_batch(self):
        F = np.array([[1.0, 2.0], [3.0, 0.5]])
        w = np.array([1.0, 1.0])
        z = np.array([0.0, 0.0])
        result = tchebycheff(F, w, z)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)


class TestAutoPartitions:
    def test_2obj(self):
        H = auto_partitions(100, 2)
        # C(H+1, 1) = H+1 <= 100 -> H <= 99
        assert H == 99

    def test_3obj_reasonable(self):
        H = auto_partitions(100, 3)
        # C(H+2, 2) <= 100; C(12+2,2)=91, C(13+2,2)=105
        assert H == 12


class TestAssociation:
    def test_each_point_gets_one_weight(self):
        w = das_dennis(9, 2)
        F = np.random.default_rng(0).random((20, 2))
        assoc = associate_to_weights(F, w)
        assert assoc.shape == (20,)
        assert np.all(assoc >= 0)
        assert np.all(assoc < w.shape[0])
