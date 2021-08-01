"""Pure GRASP solver"""
import logging
from random import sample
from timeit import default_timer
from typing import List, Optional, Tuple

import numpy as np

from python_tsp.heuristics.local_search import solve_tsp_local_search
from python_tsp.utils import compute_permutation_distance

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(level=logging.WARNING)
logger.addHandler(ch)


def solve_tsp_grasp(
    distance_matrix: np.ndarray,
    start_position: int = 0,
    alpha: float = 0.5,
    max_iterations: int = 1,
    perturbation_scheme: str = "two_opt",
    max_processing_time: Optional[float] = None,
    log_file: Optional[str] = None,
) -> Tuple[List, float]:
    """Solve a TSP problem with a Greedy Randomized Adaptive Search Procedure
    (GRASP) heuristic.

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    start_position {0}
        The row index of the distance matrix the indicates where we will start
        the GRA part, that means the Greedy Randomized Adaptive section.

    alpha {0.5}
        It can take values between 0 and 1. When it is zero, GRASP takes
        the form of a randomized algorithm. When it is 1, GRASP takes the form
        a greedy algorithm. The user should experiment with different values.

    max_iterations {1}
        The maximum number of iterations to consider using the GRASP process.
        That means using the cronstr

    perturbation_scheme {"ps1", "ps2", "ps3", "ps4", "ps5", "ps6", ["two_opt"]}
        Mechanism used to generate new solutions in the local search part.
        Defaults to "two_opt".

    max_processing_time {None}
        Maximum processing time in seconds. If not provided, the method stops
        only when a local minimum is obtained

    log_file
        If not `None`, creates a log file with details about the whole
        execution

    Returns
    -------
    A permutation of nodes from 0 to n - 1 that produces the least total
    distance obtained (not necessarily optimal).

    The total distance the returned permutation produces.

    Notes
    -----
    Here are the steps of the algorithm:
        1. In the "constructive phase", starting from an initial point,
        add another node by randomly selecting a selection of nodes that
        have g_{start_point} < g_min + \\alpha(g_max - g_min). Do that again
        considering the start_point as the latest added node. Repeat this
        process until you have the full path.
        2. If this path/tour has a lower distance than an inital random route,
        then perfom a local search in this path/tour. The resulting becomes the
        new standard in other iterations.
        3. Do this 'max iterations' times.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('Alpha must be between 0 and 1.')
    n = distance_matrix.shape[0]
    if start_position >= n or start_position < 0:
        raise ValueError('start_position must be one of the index of rows.')

    max_processing_time = max_processing_time or np.inf
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    tic = default_timer()
    best_Tour = setup(distance_matrix, None)[0]
    i = 0
    while i <= max_iterations:
        intial_Tour = constructive_phase(distance_matrix, alpha,
                                         start_position)
        optimized_Tour = solve_tsp_local_search(distance_matrix,
                                                intial_Tour,
                                                perturbation_scheme,
                                                max_processing_time)[0]

        f_best_tour = compute_permutation_distance(distance_matrix,
                                                   best_Tour)
        f_optimized_tour = compute_permutation_distance(distance_matrix,
                                                        optimized_Tour)

        if f_best_tour > f_optimized_tour:
            best_Tour = optimized_Tour
            msg = f"""Current value: {f_optimized_tour};\n
            Neighbor: {optimized_Tour}"""
            logger.info(msg)

        if default_timer() - tic > max_processing_time:
            logger.warning("Stopping early due to time constraints")
            i = max_iterations + 1
            break

        i += 1
    return best_Tour, compute_permutation_distance(distance_matrix, best_Tour)


def constructive_phase(
    distance_matrix: np.ndarray,
    alpha: float,
    start: int = 0
) -> List:
    Tour = [start]

    for _ in range(distance_matrix.shape[0] - 1):
        min_index = get_maxmin_index_from_row(distance_matrix,
                                              Tour[-1], Tour, 'min')
        max_index = get_maxmin_index_from_row(distance_matrix,
                                              Tour[-1], Tour, 'max')

        f_min = distance_matrix[Tour[-1]][min_index]
        f_max = distance_matrix[Tour[-1]][max_index]

        # List of Restrict Candidates = LRC
        LRC_index = np.array(range(len(distance_matrix[Tour[-1]])))

        LRC_condition = (
            distance_matrix[Tour[-1]] <= f_min + alpha * (f_max - f_min)
        )
        LRC_condition[Tour[-1]] = False
        LRC_index = LRC_index[LRC_condition]

        new_city_index = np.random.choice(LRC_index, 1, replace=False)[0]
        Tour.append(new_city_index)

    return Tour


def get_maxmin_index_from_row(
    distance_matrix: np.ndarray,
    row: int,
    previous_indexes: List,
    type: str,
) -> int:
    """
    Get the minimum/maximum element in the adjusted row array from
    a distance matrix. We adjust the row array in order to never get
    the "previous_indexes" list of indexes.
    """
    distance_matrix = distance_matrix.copy()
    arr = distance_matrix[row].astype(float)

    aux_list = range(arr.shape[0])
    aux_list_2 = []
    for i in aux_list:
        if i in previous_indexes:
            aux_list_2.append(True)
        else:
            aux_list_2.append(False)
    previous_indexes_bool = aux_list_2

    if type == 'max':
        arr[previous_indexes_bool] = -1
        target_index = np.argmax(arr)
    if type == 'min':
        arr[previous_indexes_bool] = np.Inf
        target_index = np.argmin(arr)

    return int(target_index)


def setup(
    distance_matrix: np.ndarray, x0: Optional[List] = None
) -> Tuple[List[int], float]:
    """Return initial solution and its objective value

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Permutation of nodes from 0 to n - 1 indicating the starting solution.
        If not provided, a random list is created.

    Returns
    -------
    x0
        Permutation with initial solution. If ``x0`` was provided, it is the
        same list

    fx0
        Objective value of x0
    """

    if not x0:
        n = distance_matrix.shape[0]  # number of nodes
        x0 = [0] + sample(range(1, n), n - 1)  # ensure 0 is the first node

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0
