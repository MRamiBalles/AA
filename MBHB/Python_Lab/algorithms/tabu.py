import numpy as np
from .base import BaseAlgorithm
from core.evaluator import calculate_fitness, calculate_delta

class TabuSearch(BaseAlgorithm):
    """
    Tabu Search (TS) for QAP.
    Memory: Tabu Matrix (iteration of unlocking).
    Tenence: n / 2
    Neighborhood: 40 random neighbors per iteration.
    Restart: Every 8n iterations, reset to random/best/greedy.
    """
    def __init__(self, n, flow, distance, initial_perm=None, seed=None):
        super().__init__(n, flow, distance, seed)
        self.initial_perm = initial_perm

    def solve(self):
        if self.initial_perm is not None:
            actual = np.copy(self.initial_perm)
        else:
            actual = np.random.permutation(self.n)
            
        coste_actual = calculate_fitness(actual, self.flow, self.distance)
        
        mejor_global = np.copy(actual)
        coste_mejor_global = coste_actual
        
        tenencia = self.n // 2
        memoria_tabu = np.zeros((self.n, self.n), dtype=int)
        
        iteraciones_totales = 100 * self.n
        reinicio_interval = 8 * self.n
        
        evaluations = 1
        history = [(evaluations, coste_mejor_global)]
        
        for iter_count in range(1, iteraciones_totales + 1):
            
            mejor_delta_vecindario = float('inf')
            move_r, move_s = -1, -1
            
            # Examine 40 random neighbors
            for _ in range(40):
                r, s = np.random.choice(self.n, 2, replace=False)
                
                evaluations += 1
                delta = calculate_delta(actual, self.flow, self.distance, r, s)
                coste_vecino = coste_actual + delta
                
                is_tabu = memoria_tabu[r, s] > iter_count
                
                # Aspiration Criterion
                if is_tabu and coste_vecino < coste_mejor_global:
                    is_tabu = False
                
                if not is_tabu:
                    if delta < mejor_delta_vecindario:
                        mejor_delta_vecindario = delta
                        move_r, move_s = r, s
            
            # Apply Best Non-Tabu Move
            if move_r != -1:
                actual[move_r], actual[move_s] = actual[move_s], actual[move_r]
                coste_actual += mejor_delta_vecindario
                
                # Update Tabu Memory
                memoria_tabu[move_r, move_s] = iter_count + tenencia
                memoria_tabu[move_s, move_r] = iter_count + tenencia
                
                if coste_actual < coste_mejor_global:
                    mejor_global = np.copy(actual)
                    coste_mejor_global = coste_actual
                    
            # Record best cost over evaluation history
            history.append((evaluations, coste_mejor_global))
            
            # Restart Policy (Diversification)
            if iter_count % reinicio_interval == 0:
                actual = np.random.permutation(self.n)
                coste_actual = calculate_fitness(actual, self.flow, self.distance)
                memoria_tabu.fill(0)
                
        return mejor_global, coste_mejor_global, history
