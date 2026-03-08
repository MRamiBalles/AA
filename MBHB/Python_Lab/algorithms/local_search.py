import numpy as np
from .base import BaseAlgorithm
from core.evaluator import calculate_fitness, calculate_delta

class LocalSearch(BaseAlgorithm):
    """
    Local Search implementation for QAP.
    Strategy: First Improvement with 2-exchange neighborhood.
    """
    def __init__(self, n, flow, distance, initial_perm=None, seed=None):
        super().__init__(n, flow, distance, seed)
        self.initial_perm = initial_perm

    def solve(self):
        # Start with initial permutation or random
        if self.initial_perm is not None:
            current_perm = np.copy(self.initial_perm)
        else:
            current_perm = np.random.permutation(self.n)
            
        current_cost = calculate_fitness(current_perm, self.flow, self.distance)
        
        improved = True
        evaluations = 1 
        history = [(evaluations, current_cost)]
        
        while improved:
            improved = False
            # Generate random search order for neighborhood to avoid bias
            indices = np.random.permutation(self.n)
            
            for i_idx in range(self.n):
                i = indices[i_idx]
                for j_idx in range(i_idx + 1, self.n):
                    j = indices[j_idx]
                    
                    evaluations += 1
                    
                    # Optimized change calculation O(n)
                    delta = calculate_delta(current_perm, self.flow, self.distance, i, j)
                    
                    if delta < 0:
                        # Improvement found (First Improvement strategy)
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                        current_cost += delta
                        improved = True
                        history.append((evaluations, current_cost))
                        break # Break inner loop
                if improved:
                    break # Break outer loop
                    
        return current_perm, current_cost, history
