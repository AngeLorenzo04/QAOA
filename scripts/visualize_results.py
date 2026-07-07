import json
import os
import sys

# Aggiunge src al path per importare plotter.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.visualization.plotter import plot_approximation_ratio_vs_params, plot_optimizer_convergence, plot_probability_distribution

SUMMARY_FILE = "data/benchmarking_results/qaoa_benchmarking_summary.json"

def main():
    if not os.path.exists(SUMMARY_FILE):
        print(f"File {SUMMARY_FILE} non trovato. Assicurati che il benchmarking abbia finito di salvare i risultati.")
        return

    with open(SUMMARY_FILE, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("Il file dei risultati è vuoto.")
        return

    print("Generazione grafico: Approximation Ratio vs Numero di Vertici (N)...")
    plot_approximation_ratio_vs_params(
        data, 
        'n_vertices', 
        "Approximation Ratio vs Numero di Nodi",
        filepath="data/benchmarking_results/plot_approx_vs_N.png"
    )

    print("Generazione grafico: Approximation Ratio vs Profondità del circuito (p)...")
    plot_approximation_ratio_vs_params(
        data, 
        'p_value', 
        "Approximation Ratio vs Layer p",
        filepath="data/benchmarking_results/plot_approx_vs_p.png"
    )

    print("Generazione grafico: Approximation Ratio vs Densità...")
    plot_approximation_ratio_vs_params(
        data, 
        'density_edges', 
        "Approximation Ratio vs Densità Archi",
        filepath="data/benchmarking_results/plot_approx_vs_density.png"
    )

    # Prendi un risultato a campione per visualizzare la convergenza dell'ottimizzatore
    sample_result = data[0]
    n = sample_result['graph_metadata']['n_vertices']
    p = sample_result['qaoa_config']['p_value']
    
    print(f"Generazione grafico: Convergenza Ottimizzatore (Campione: N={n}, p={p})...")
    plot_optimizer_convergence(
        sample_result['metrics']['optimization_history'], 
        f"Convergenza Ottimizzatore (N={n}, p={p})",
        filepath="data/benchmarking_results/plot_convergence_sample.png"
    )
    
    print("\nTutte le dashboard sono state generate e salvate in 'data/benchmarking_results/'!")

if __name__ == "__main__":
    main()
