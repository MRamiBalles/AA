
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    """
    Loads data from CSV.
    Assumes column 0 is population (x) and column 1 is profit (y).
    """
    data = np.loadtxt(path, delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y

def plot_data(X, y):
    """
    Plots the data points.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, marker='x', c='r', label='Training Data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.legend()
    plt.grid(True)
    plt.savefig('scatter_plot.png')
    # plt.show() # Commented out for headless environment

def compute_cost(X, y, theta):
    """
    Computes the cost J(theta) for linear regression using the vectorized formula.
    J(theta) = (1 / 2m) * (X * theta - y)^T * (X * theta - y)
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        
        # Vectorized gradient update:
        # theta = theta - (alpha / m) * (X^T * errors)
        delta = (1 / m) * (X.T.dot(errors))
        theta = theta - alpha * delta
        
        J_history[i] = compute_cost(X, y, theta)
        
    return theta, J_history

def main():
    print("Loading Data...")
    X_raw, y = load_data('regresion_data.csv')
    m = len(y)
    
    print("Plotting Data...")
    plot_data(X_raw, y)
    
    # Add intercept term to X (column of ones)
    print("Preparing Matrices...")
    X = np.stack([np.ones(m), X_raw], axis=1)
    
    # Initialize theta
    theta = np.zeros(2)
    
    # Gradient Descent Settings
    iterations = 1500
    alpha = 0.01
    
    print(f"Initial Cost J(0,0): {compute_cost(X, y, theta):.2f}")
    
    # Run Gradient Descent
    print("Running Gradient Descent...")
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    
    print(f"Theta found by gradient descent: {theta}")
    print(f"Final Cost: {compute_cost(X, y, theta):.4f}")
    
    # Plot Linear Fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X_raw, y, marker='x', c='r', label='Training Data')
    plt.plot(X_raw, X.dot(theta), label='Linear Regression', c='b')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.legend()
    plt.savefig('linear_fit.png')
    
    # Predict for specific populations (Manual check)
    predict1 = np.array([1, 3.5]).dot(theta)
    predict2 = np.array([1, 7.0]).dot(theta)
    print(f"For population = 35,000, we predict a profit of {predict1 * 10000:.2f}")
    print(f"For population = 70,000, we predict a profit of {predict2 * 10000:.2f}")

if __name__ == "__main__":
    main()
