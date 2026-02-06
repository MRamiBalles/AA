import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def plot_decision_boundary(clf, X, y, title, filename):
    """
    Plots the decision boundary, margins, and support vectors.
    """
    plt.figure(figsize=(8, 6))
    
    # Create meshgrid to plot decision function
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contours (Decision Boundary and Margins)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    
    # Highlight Support Vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
                linewidth=2, facecolors='none', edgecolors='k', label='Support Vectors')
    
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    # plt.show() # Commented for headless execution
    print(f"Saved plot: {filename}")

def main():
    # --- Part 1: Verification of Manual Exercise ---
    print("Running Part 1: Manual Exercise Verification...")
    X_manual = np.array([[2, 0], [0, 2], [0, 0]])
    y_manual = np.array([1, 1, -1])
    
    # Large C to enforce Hard Margin (as in manual calculation)
    clf_manual = svm.SVC(kernel='linear', C=1000)
    clf_manual.fit(X_manual, y_manual)
    
    print(f"Manual Exercise Weights (w): {clf_manual.coef_[0]}")
    print(f"Manual Exercise Bias (b): {clf_manual.intercept_[0]}")
    print(f"Support Vectors: \n{clf_manual.support_vectors_}")
    
    plot_decision_boundary(clf_manual, X_manual, y_manual, 
                           "Part 1: Manual Exercise (Hard Margin)", 
                           "svm_part1_manual.png")

    # --- Part 2: Linear SVM on Blobs ---
    print("\nRunning Part 2: Linear SVM on Blobs...")
    X_blobs, y_blobs = datasets.make_blobs(n_samples=50, centers=2, random_state=6)
    
    clf_linear = svm.SVC(kernel='linear', C=1.0)
    clf_linear.fit(X_blobs, y_blobs)
    
    plot_decision_boundary(clf_linear, X_blobs, y_blobs, 
                           "Part 2: Linear SVM (Blobs)", 
                           "svm_part2_linear.png")

    # --- Part 3: Non-Linear SVM (RBF) on Circles ---
    print("\nRunning Part 3: Non-Linear SVM (RBF) on Circles...")
    X_circles, y_circles = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1)
    
    # RBF Kernel (Radial Basis Function)
    clf_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf_rbf.fit(X_circles, y_circles)
    
    plot_decision_boundary(clf_rbf, X_circles, y_circles, 
                           "Part 3: RBF SVM (Circles)", 
                           "svm_part3_rbf.png")

if __name__ == "__main__":
    main()
