import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def plot_decision_boundary(clf, X, y, filename):
    """
    Plots decision boundary for XOR problem.
    """
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=100)
    plt.title("MLP Decision Boundary (XOR Problem)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

def main():
    # --- Part 1: Solving XOR (Non-linear Problem) ---
    print("--- Part 1: XOR Problem ---")
    # XOR Logic Gate: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print("Training MLP on XOR data...")
    # Architecture: 2 Inputs -> Hidden Layer (4 neurons, ReLU) -> 1 Output (Sigmoid/Logistic)
    # solver='lbfgs' is better for small datasets
    clf_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='lbfgs', 
                            random_state=1, max_iter=2000)
    
    clf_xor.fit(X_xor, y_xor)
    
    print("Predictions:", clf_xor.predict(X_xor))
    print("Score:", clf_xor.score(X_xor, y_xor))
    
    plot_decision_boundary(clf_xor, X_xor, y_xor, "mlp_xor_boundary.png")
    
    # --- Part 2: Handwritten Digits Classification ---
    print("\n--- Part 2: Digits Classification (MNIST Lite) ---")
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset Shape: {X.shape} (N_samples, N_features)")
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training MLP on Digits...")
    # Architecture: Input(64) -> Hidden(30) -> Hidden(15) -> Output(10)
    clf_digits = MLPClassifier(hidden_layer_sizes=(30, 15), activation='relu', solver='adam',
                               max_iter=1000, random_state=42)
    
    clf_digits.fit(X_train, y_train)
    
    # Evaluation
    predictions = clf_digits.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(clf_digits.loss_curve_)
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("mlp_loss_curve.png")
    print("Saved plot: mlp_loss_curve.png")

if __name__ == "__main__":
    main()
