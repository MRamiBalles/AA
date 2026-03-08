import numpy as np
from .base import BaseAlgorithm

class GreedyAlgorithm(BaseAlgorithm):
    """
    Greedy implementation for QAP.
    Strategy: Assign facilities with high total flow to locations with low total distance.
    """
    def solve(self):
        # Calculate total flow for each facility (Sum rows + cols as in C++)
        total_flow = np.sum(self.flow, axis=0) + np.sum(self.flow, axis=1)
        # Calculate total distance for each location (Sum rows + cols as in C++)
        total_dist = np.sum(self.distance, axis=0) + np.sum(self.distance, axis=1)
        
        # Sort facilities by descending flow (id and potential)
        fac_indices = np.argsort(-total_flow)
        # Sort locations by ascending distance (id and potential)
        loc_indices = np.argsort(total_dist)
        
        # Create permutation: index is facility, value is location
        permutation = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            permutation[fac_indices[i]] = loc_indices[i]
            
        from core.evaluator import calculate_fitness
        cost = calculate_fitness(permutation, self.flow, self.distance)
        
        return permutation, cost
