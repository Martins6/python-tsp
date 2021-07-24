import unittest
from python_tsp.distances import tsplib_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search


class TestGRASP(unittest.TestCase):
    def test_grasp(self):
        tsplib_file = "tests/tsplib_data/a280.tsp"
        distance_matrix = tsplib_distance_matrix(tsplib_file)
        