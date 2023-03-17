import numpy as np
from enum import Enum


class BoundaryConditions(Enum):
    NotAKnot = 0
    Natural = 1
    Clamped = 2


def cubic(points: list[tuple[int, int]], boundary_condition=BoundaryConditions.Natural) -> np.ndarray:
    """Interpolate a set of points with cubic splines.

    Args:
        points (list[tuple[int, int]]): A list of points to interpolate.
        boundary_condition (BoundaryConditions, optional): The boundary condition to use. Defaults to BoundaryConditions.Natural.

    Returns:
        np.ndarray: A matrix representing the coefficients of the splines.
    """

    num_points = len(points)
    num_splines = num_points - 1

    system = np.zeros((num_splines * 3, num_splines * 3), dtype=np.float64)
    rhs = np.zeros(num_splines * 3, dtype=np.float64)
    a_terms = np.zeros(num_splines, dtype=np.float64)

    for spline_index, (start_point, end_point) \
            in enumerate(zip(points[:-1], points[1:])):
        start_x, start_y = start_point
        end_x, end_y = end_point
        offset = spline_index * 3

        # Splines must be continuous
        a_terms[spline_index] = start_y

        system[offset, offset:offset + 3] = [end_x -
                                             start_x, (end_x-start_x)**2, (end_x-start_x)**3]
        rhs[offset] = end_y - start_y
        if spline_index < num_splines - 1:
            # Splines must be differentiable (1st derivative)
            system[offset + 1, offset:offset + 3] = [1,
                                                     2 * (end_x-start_x), 3 * (end_x-start_x)**2]
            system[offset + 1, offset + 3:offset + 6] = [-1, 0, 0]
            rhs[offset + 1] = 0

            # Splines must be differentiable (2nd derivative)
            system[offset + 2, offset:offset +
                   3] = [0, 2, 6 * (end_x-start_x)]
            system[offset + 2, offset + 3:offset + 6] = [0, -2, 0]
            rhs[offset + 2] = 0
        else:
            match (boundary_condition):
                case BoundaryConditions.Natural:
                    # Second derivative at end conditions is 0s
                    system[offset + 1, 0:3] = [0, 2, 0]
                    rhs[offset + 1] = 0

                    system[offset + 2, offset:offset +
                           3] = [0, 2, 6 * (end_x-start_x)]
                    rhs[offset + 2] = 0

                case _:
                    raise NotImplementedError(
                        "Only natural boundary conditions are supported")
                    # Use https://en.wikipedia.org/wiki/Spline_interpolation#Introduction to find the rest
    print(system, rhs)
    solution = gaussian_elimination(system, rhs)
    return np.concatenate((a_terms[:, None], solution.reshape((num_splines, 3))), axis=1)


def gaussian_elimination(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a system of linear equations using Gaussian elimination.

    Args:
        matrix (np.ndarray): A matrix representing the system of equations.
        rhs (np.ndarray): The right hand side of the system of equations.

    Returns:
        np.ndarray: The solution to the system of equations.
    """

    augmented_matrix = np.concatenate((matrix, rhs[:, None]), axis=1)
    num_rows, num_cols = augmented_matrix.shape

    for row_index in range(num_rows):
        # Find the largest pivot, and swap rows if necessary
        pivot_row = np.argmax(
            np.abs(augmented_matrix[row_index:, row_index])) + row_index
        augmented_matrix[[row_index, pivot_row]] = augmented_matrix[
            [pivot_row, row_index]]

        # Eliminate the column below to get to row echelon form
        for row in range(row_index + 1, num_rows):
            # Get the multiplier, and subtract the row from the current row
            # This will eliminate the column below, no matter what the pivot is
            augmented_matrix[row] -= augmented_matrix[row_index] * \
                augmented_matrix[row, row_index] / \
                augmented_matrix[row_index, row_index]

    # Backsubstitution
    solution = np.zeros(num_rows)
    for row_index in range(num_rows - 1, -1, -1):
        solution[row_index] = augmented_matrix[row_index, -1] / \
            augmented_matrix[row_index, row_index]

        augmented_matrix[:, row_index] = augmented_matrix[:,
                                                          row_index] * solution[row_index]
        augmented_matrix[:, -1] = augmented_matrix[:, -1] - augmented_matrix[:, row_index]
    return solution


def tests():
    assert gaussian_elimination(np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 3, -1, 0, 0],
        [0, 3, 0, -1, 0],
        [0, 0, 0, 2, 6]
    ], dtype=np.float64), np.array([1, 2, 0, 0, 0], dtype=np.float64)).all() == np.array([3/4, 3/2, 3/4, 1/4, -1/4], dtype=np.float64).all()
    

if __name__ == "__main__":
    tests()
    print(cubic([(1, 2), (2, 3), (3, 5)]))
