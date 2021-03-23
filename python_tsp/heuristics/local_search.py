"""Simple local search solver"""
from random import sample
from timeit import default_timer
from typing import List, Optional, Tuple

import numpy as np

from python_tsp.utils import compute_permutation_distance
from python_tsp.heuristics.perturbation_schemes import neighborhood_gen


def solve_tsp_local_search(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    perturbation_scheme: str = "two_opt",
    max_processing_time: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[List, float]:
    """Solve a TSP problem with a local search heuristic

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

    verbose
        `True` to display information about the process

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

    tic = default_timer()
    stop_early = False
    improvement = True
    while improvement and (not stop_early):
        improvement = False
        for n_index, xn in enumerate(neighborhood_gen[perturbation_scheme](x)):
            if default_timer() - tic > max_processing_time:
                if verbose:
                    print("\nStopping early due to time constraints")
                stop_early = True
                break

            fn = compute_permutation_distance(distance_matrix, xn)

            if verbose:
                print(f"Current value: {fx}; Neighbor: {n_index}", end="\r")

            if fn < fx:
                improvement = True
                x, fx = xn, fn
                break  # early stop due to first improvement local search

        if verbose:
            print("")  # line break

    return x, fx


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
