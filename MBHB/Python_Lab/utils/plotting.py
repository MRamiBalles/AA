import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_algorithm_comparison(results_df, output_dir="results"):
    """
    Generates professional comparison plots from a results DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Solution Cost Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="algorithm", y="cost", data=results_df, palette="viridis", hue="algorithm", legend=False)
    plt.title("QAP Solution Quality Comparison (Lower is Better)")
    plt.ylabel("Total Cost (Fitness)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_comparison.png"))
    plt.close()
    
    # Plot 2: Execution Time Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="algorithm", y="time_seconds", data=results_df, palette="magma", hue="algorithm", legend=False)
    plt.title("Algorithm Execution Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()

def plot_convergence(results_df, output_dir="results"):
    """
    Generates convergence plots (Cost vs Evaluations) for each algorithm and a master comparison plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 8))
    
    # Extract the first run for each algorithm for cleaner visualization
    for idx, algo in enumerate(results_df['algorithm'].unique()):
        # Get the first run's history for this algorithm
        run_data = results_df[results_df['algorithm'] == algo].iloc[0]
        history = run_data['history']
        
        if not history:
            continue
            
        evals = [h[0] for h in history]
        costs = [h[1] for h in history]
        
        # Plot individual convergence
        plt.figure(figsize=(8, 5))
        plt.plot(evals, costs, marker='o', markersize=3, linestyle='-', linewidth=2, color=sns.color_palette("tab10")[idx])
        plt.title(f"Convergence Graph: {algo}")
        plt.xlabel("Evaluations (Iteraciones Equivalentes)")
        plt.ylabel("Mejor Coste Encontrado")
        max_cost, min_cost = max(costs), min(costs)
        # Add some padding to Y axis
        plt.ylim(min_cost * 0.95, max_cost * 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"convergence_{algo}.png"))
        plt.close()
        
    # Plot Master Convergence (All in one)
    plt.figure(figsize=(12, 8))
    for algo in results_df['algorithm'].unique():
        run_data = results_df[results_df['algorithm'] == algo].iloc[0]
        history = run_data['history']
        
        if not history:
            continue
            
        evals = [h[0] for h in history]
        costs = [h[1] for h in history]
        plt.plot(evals, costs, label=algo, linewidth=2)
        
    plt.title("Master Convergence Comparison (Cost vs Evaluations)")
    plt.xlabel("Evaluations (Iteraciones)")
    plt.ylabel("Mejor Coste")
    
    # SA and Tabu have many evaluations, Greedy/LS have very few.
    # Use log scale on X to make the graph readable for all
    plt.xscale('log') 
    
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_master.png"))
    plt.close()

def plot_language_comparison(lang_data, output_dir="results"):
    """
    Generates a chart comparing Python vs C++ performance (manually provided data or detected).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(lang_data)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x="language", y="time", data=df, palette="cool", hue="language", legend=False)
    plt.title("Cross-Language Execution Comparison (Python vs C++)")
    plt.ylabel("Execution Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "language_comparison.png"))
    plt.close()

    """
    Generates a chart comparing Python vs C++ performance (manually provided data or detected).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(lang_data)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x="language", y="time", data=df, palette="cool")
    plt.title("Cross-Language Execution Comparison (Python vs C++)")
    plt.ylabel("Execution Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "language_comparison.png"))
    plt.close()
