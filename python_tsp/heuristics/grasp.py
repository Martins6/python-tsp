"""Pure GRASP solver"""
import logging
from random import sample
from timeit import default_timer
from typing import List, Optional, Tuple, Callable

import numpy as np

from python_tsp.utils import compute_permutation_distance
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen
from python_tsp.heuristics.local_search import solve_tsp_local_search

# Testing
from python_tsp.distances import tsplib_distance_matrix


logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(level=logging.WARNING)
logger.addHandler(ch)


def solve_tsp_grasp(
    distance_matrix: np.ndarray,
    start_position: int = 0,
    alpha: float = 0.5,
    perturbation_scheme: str = "two_opt",
    max_iterations: int = 100, 
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
    x, fx = setup(distance_matrix, x0)
    max_processing_time = max_processing_time or np.inf
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    tic = default_timer()
    
    best_Tour = []
    i = 0
    while i <= max_iterations:
        constructive_phase(distance_matrix, alpha, compute_permutation_distance)
        
        
        
        i + 1
    
    # stop_early = False
    # improvement = True
    # while improvement and (not stop_early):
    #     improvement = False
    #     for n_index, xn in enumerate(neighborhood_gen[perturbation_scheme](x)):
    #         if default_timer() - tic > max_processing_time:
    #             logger.warning("Stopping early due to time constraints")
    #             stop_early = True
    #             break

    #         fn = compute_permutation_distance(distance_matrix, xn)
    #         logger.info(f"Current value: {fx}; Neighbor: {n_index}")

    #         if fn < fx:
    #             improvement = True
    #             x, fx = xn, fn
    #             break  # early stop due to first improvement local search
    
    return x, fx

def constructive_phase(
    distance_matrix: np.ndarray,
    alpha: float,
    objective_function: Callable,
    start: int = 0
) -> List:    
    Tour = [start]
    
    for _ in range(distance_matrix.shape[0] - 1):
        min_index = get_maxmin_index_from_row(distance_matrix, Tour[-1], Tour[0:-1], 'min')
        max_index = get_maxmin_index_from_row(distance_matrix, Tour[-1], Tour[0:-1], 'max')
    
    # We must always return to the same city.
    Tour.append(start)

    return np.array(Tour)

def get_maxmin_index_from_row(
    distance_matrix: np.ndarray,
    row: int,
    previous_indexes: List,
    type: str,
    )-> int:
    """Get the minimum/maximum element in the adjusted row array from a distance matrix.
    We adjust the row array in order to never get the "previous_indexes" list of indexes.
    """
    distance_matrix = distance_matrix.copy()
    arr = distance_matrix[row]
    if type == 'max':
        arr[previous_indexes] = -1
        target_index = np.argmax(distance_matrix[row])
    if type == 'min':
        arr[previous_indexes] = np.Inf
        target_index = np.argmin(distance_matrix[row])
    target_index = target_index[0] if len(target_index) > 1 else target_index
    print(target_index)
    
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


if __name__ == '__main__':
    tsplib_file = "tests/tsplib_data/a280.tsp"
    distance_matrix = tsplib_distance_matrix(tsplib_file)
    print(distance_matrix)
        