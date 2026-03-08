from abc import ABC, abstractmethod
import time
import numpy as np

class BaseAlgorithm(ABC):
    """
    Abstract base class for QAP metaheuristics.
    """
    def __init__(self, n, flow, distance, seed=None):
        self.n = n
        self.flow = flow
        self.distance = distance
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    @abstractmethod
    def solve(self):
        """
        Executes the algorithm and returns the best solution and its cost.
        Returns:
            best_permutation (np.ndarray)
            best_cost (int)
            execution_time (float)
        """
        pass

    def run(self):
        """
        Wrapper to measure execution time.
        """
        start_time = time.time()
        best_perm, best_cost = self.solve()
        end_time = time.time()
        return best_perm, best_cost, end_time - start_time
