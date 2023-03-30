import pytest
from splines import cubic, BoundaryConditions, gaussian_elimination
import numpy as np


def test_only_natural_splines():
    with pytest.raises(NotImplementedError):
        cubic([(0, 0), (1, 1)], boundary_condition=BoundaryConditions.Clamped)

    with pytest.raises(NotImplementedError):
        cubic([(0, 0), (1, 1)], boundary_condition=BoundaryConditions.NotAKnot)


def test_gaussian_elimination():
    assert gaussian_elimination(np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 3, -1, 0, 0],
        [0, 3, 0, -1, 0],
        [0, 0, 0, 2, 6]
    ], dtype=np.float64), np.array([1, 2, 0, 0, 0], dtype=np.float64)).all() == np.array([3/4, 3/2, 3/4, 1/4, -1/4], dtype=np.float64).all()
