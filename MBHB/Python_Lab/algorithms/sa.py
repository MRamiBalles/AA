import numpy as np
import math
from .base import BaseAlgorithm
from core.evaluator import calculate_fitness, calculate_delta

class SimulatedAnnealing(BaseAlgorithm):
    """
    Simulated Annealing (SA) for QAP.
    Cooling: Cauchy (T_k = T_0 / (1 + k))
    T0: Initialized based on mu=0.3, phi=0.3
    Markov Chain: Limit to 40 neighbors or 5 successes per temperature.
    """
    def __init__(self, n, flow, distance, initial_perm=None, seed=None):
        super().__init__(n, flow, distance, seed)
        self.initial_perm = initial_perm
        self.mu = 0.3
        self.phi = 0.3

    def solve(self):
        if self.initial_perm is not None:
            actual = np.copy(self.initial_perm)
        else:
            actual = np.random.permutation(self.n)
            
        coste_actual = calculate_fitness(actual, self.flow, self.distance)
        
        mejor = np.copy(actual)
        coste_mejor = coste_actual
        
        # Calculate T0
        # (MU / -log(PHI)) * costeInicial
        t0 = (self.mu / -math.log(self.phi)) * coste_actual
        t = t0
        max_enfriamientos = 50 * self.n
        
        evaluations = 1
        history = [(evaluations, coste_mejor)]
        
        for k in range(max_enfriamientos):
            exitos = 0
            vecinos_generados = 0
            
            while vecinos_generados < 40 and exitos < 5:
                vecinos_generados += 1
                evaluations += 1
                
                # Generate random swap
                r, s = np.random.choice(self.n, 2, replace=False)
                
                delta = calculate_delta(actual, self.flow, self.distance, r, s)
                
                # Metropolis Criterion
                if delta < 0 or np.random.random() < math.exp(-delta / t):
                    actual[r], actual[s] = actual[s], actual[r]
                    coste_actual += delta
                    exitos += 1
                    
                    if coste_actual < coste_mejor:
                        mejor = np.copy(actual)
                        coste_mejor = coste_actual
                        
                # Record convergence of the best found so far
                history.append((evaluations, coste_mejor))
            
            # Cauchy Cooling
            t = t0 / (1.0 + (k + 1))
            
            if t < 0.001:
                break
                
        return mejor, coste_mejor, history
