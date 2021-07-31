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
    perturbation_scheme: str = "two_opt",
    max_iterations: int = 1,
    max_processing_time: Optional[float] = None,
    log_file: Optional[str] = None,
) -> Tuple[List, float]:
    """Solve a TSP problem with a GRASP heuristic

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path

    perturbation_scheme {"ps1", "ps2", "ps3", "ps4", "ps5", "ps6", ["two_opt"]}
        Mechanism used to generate new solutions. Defaults to "two_opt"

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
        1. Let `x`, `fx` be a initial solution permutation and its objective
        value;
        2. Perform a neighborhood search in `x`:
            2.1 For each `x'` neighbor of `x`, if `fx'` < `fx`, set `x` <- `x'`
            and stop;
        3. Repeat step 2 until all neighbors of `x` are tried and there is no
        improvement. Return `x`, `fx` as solution.
    """
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
    return best_Tour, compute_permutation_distance(best_Tour, distance_matrix)


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
            distance_matrix[Tour[-1]] <= f_min + alpha*(f_max - f_min)
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
    
    return target_index


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

    
    
        