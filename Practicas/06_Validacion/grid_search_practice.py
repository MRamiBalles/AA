
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def main():
    print("--- Bonus Practice: Hyperparameter Tuning (Grid Search) ---")
    
    # 1. Load Data
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    print(f"Dataset shape: {X.shape}")
    
    # 2. Preprocessing (Scaling is crucial for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 4. Define Parameter Grid
    # C: Regularization parameter (Trade-off between smooth boundary and correct classification)
    # gamma: Kernel coefficient for 'rbf' (Influence of single training example)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    
    print("\nStarting Grid Search with Cross-Validation (cv=5)...")
    print(f"Testing parameters: {param_grid}")
    
    # 5. Grid Search
    # refit=True ensures the model is retrained on the full training set with the best params
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)
    
    # 6. Results
    print(f"\nBest Parameters found: {grid.best_params_}")
    print(f"Best Variance Score (Accuracy): {grid.best_score_:.4f}")
    
    # 7. Evaluate on Test Set using the best model
    print("\nEvaluating Best Model on Test Set...")
    grid_predictions = grid.predict(X_test)
    
    print(classification_report(y_test, grid_predictions))
    
    # 8. Visualization of Results (Heatmap-like)
    print("Visualizing Grid Search Results...")
    results_df = pd.DataFrame(grid.cv_results_)
    pivot_table = results_df.pivot(index='param_gamma', columns='param_C', values='mean_test_score')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_table, interpolation='nearest', cmap='viridis')
    plt.title('Grid Search Accuracy (Gamma vs C)')
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.colorbar(label='Accuracy')
    
    # Set ticks
    plt.xticks(np.arange(len(param_grid['C'])), param_grid['C'])
    plt.yticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=90)
    
    plt.savefig('grid_search_heatmap.png')
    print("Saved plot: grid_search_heatmap.png")

if __name__ == "__main__":
    main()
