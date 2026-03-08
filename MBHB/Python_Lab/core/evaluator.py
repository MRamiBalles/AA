import numpy as np

def calculate_fitness(permutation, flow, distance):
    """
    Calculates the cost of a given assignment (permutation).
    cost = sum_{i=0}^{n-1} sum_{j=0}^{n-1} flow[i][j] * distance[permutation[i]][permutation[j]]
    
    Args:
        permutation (np.ndarray): Array where index is facility and value is location.
        flow (np.ndarray): Flow matrix.
        distance (np.ndarray): Distance matrix.
    
    Returns:
        int: Total cost.
    """
    n = len(permutation)
    # Using numpy indexing for potentially higher performance on larger instances
    # distance[permutation][:, permutation] creates a reordered distance matrix
    # corresponding to the facility assignments.
    
    # cost = np.sum(flow * distance[permutation][:, permutation])
    
    # Traditional double loop implementation for clarity and comparison
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += flow[i][j] * distance[permutation[i]][permutation[j]]
            
    return cost

def calculate_delta(permutation, flow, distance, r, s):
    """
    Calculates the change in cost if we swap elements at index r and s in the permutation.
    This is critical for Local Search performance (Complexity O(n) instead of O(n^2)).
    """
    n = len(permutation)
    delta = 0
    
    # Basic delta calculation for QAP (simplified for non-diagonal symmetric matrices)
    # Note: In most QAPLIB instances, flow and distance are symmetric and diagonals are 0.
    
    for k in range(n):
        if k != r and k != s:
            delta += (flow[r][k] + flow[k][r]) * (distance[permutation[s]][permutation[k]] - distance[permutation[r]][permutation[k]]) + \
                     (flow[s][k] + flow[k][s]) * (distance[permutation[r]][permutation[k]] - distance[permutation[s]][permutation[k]])
                     
    return delta

if __name__ == "__main__":
    from loader import load_qap_instance
    
    n, flow, dist = load_qap_instance("d:/AA-1/algorithm-alchemy-studio/nug5.dat")
    # Optimal permutation for nug5 is often documented, but let's test a simple one
    perm = np.arange(n)
    cost = calculate_fitness(perm, flow, dist)
    print(f"Cost of [0, 1, 2, 3, 4]: {cost}")
