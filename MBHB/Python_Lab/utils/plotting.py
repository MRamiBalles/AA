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
    ax = sns.barplot(x="algorithm", y="cost", data=results_df, palette="viridis")
    plt.title("QAP Solution Quality Comparison (Lower is Better)")
    plt.ylabel("Total Cost (Fitness)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_comparison.png"))
    plt.close()
    
    # Plot 2: Execution Time Comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="algorithm", y="time_seconds", data=results_df, palette="magma")
    plt.title("Algorithm Execution Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()

def plot_language_comparison(lang_data, output_dir="results"):
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
