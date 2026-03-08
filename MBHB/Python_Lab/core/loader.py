import os
import numpy as np

def load_qap_instance(file_path):
    """
    Loads a QAP instance from a .dat file.
    Returns:
        n (int): Dimension of the problem.
        flow (np.ndarray): Flow matrix.
        distance (np.ndarray): Distance matrix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        data = f.read().split()

    if not data:
        raise ValueError(f"File is empty: {file_path}")

    n = int(data[0])
    
    # The next n*n elements are the flow matrix
    flow = np.array(data[1:1+n*n], dtype=int).reshape((n, n))
    
    # The following n*n elements are the distance matrix
    distance = np.array(data[1+n*n:1+2*n*n], dtype=int).reshape((n, n))
    
    return n, flow, distance

if __name__ == "__main__":
    # Test loading
    try:
        n, flow, dist = load_qap_instance("d:/AA-1/algorithm-alchemy-studio/nug5.dat")
        print(f"Dimension: {n}")
        print("Flow Matrix:")
        print(flow)
        print("Distance Matrix:")
        print(dist)
    except Exception as e:
        print(f"Error: {e}")
