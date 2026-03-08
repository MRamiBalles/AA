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
        Executes the algorithm and returns the best solution, its cost, and convergence history.
        Returns:
            best_permutation (np.ndarray)
            best_cost (int)
            history (list of tuples): [(eval_count, best_cost_at_eval), ...]
        """
        pass

    def run(self):
        """
        Wrapper to measure execution time.
        """
        start_time = time.time()
        best_perm, best_cost, history = self.solve()
        end_time = time.time()
        return best_perm, best_cost, end_time - start_time, history
