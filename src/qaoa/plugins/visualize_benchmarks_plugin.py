import os
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from rich.prompt import Prompt

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.visualization.plotter import plot_approximation_ratio_vs_params

BENCHMARK_RESULTS_DIR = "data/benchmarking_results"

class VisualizeBenchmarksPlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "visualize_benchmarks"

    @property
    def description(self) -> str:
        return "Visualizza grafici di analisi e confronto dei risultati di benchmark"

    @property
    def requires_graph(self) -> bool:
        return False

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        summary_filepath = os.path.join(BENCHMARK_RESULTS_DIR, "qaoa_benchmarking_summary.json")
        if not os.path.exists(summary_filepath):
            console.print(f"[bold red]Errore: Nessun file di benchmark trovato in '{summary_filepath}'.[/bold red]")
            console.print("[yellow]Esegui prima il benchmark usando il plugin 'benchmarking'.[/yellow]")
            return

        with open(summary_filepath, 'r') as f:
            qaoa_results = json.load(f)

        if not qaoa_results:
            console.print("[bold red]Errore: Il file di benchmark è vuoto.[/bold red]")
            return

        while True:
            console.print("\n[bold purple]=== SCELTA ANALISI BENCHMARK ===[/bold purple]")
            console.print("  [bold green]1[/bold green]: Approximation Ratio vs Numero di Vertici (N)")
            console.print("  [bold green]2[/bold green]: Approximation Ratio vs Densità degli Archi (D)")
            console.print("  [bold green]3[/bold green]: Approximation Ratio vs Layer QAOA (p)")
            console.print("  [bold green]4[/bold green]: Confronto delle performance degli Ottimizzatori")
            console.print("  [bold red]i[/bold red]: Torna al menu principale")
            
            choice = Prompt.ask("Seleziona il grafico da generare", choices=["1", "2", "3", "4", "i"], default="1")
            
            if choice == "i":
                break
                
            plt.style.use('seaborn-v0_8-whitegrid')
            
            if choice == "1":
                plot_approximation_ratio_vs_params(
                    qaoa_results,
                    x_axis_param='n_vertices',
                    title="Approximation Ratio Medio vs. Numero di Vertici (N)"
                )
            elif choice == "2":
                plot_approximation_ratio_vs_params(
                    qaoa_results,
                    x_axis_param='density_edges',
                    title="Approximation Ratio Medio vs. Densità degli Archi (D)"
                )
            elif choice == "3":
                plot_approximation_ratio_vs_params(
                    qaoa_results,
                    x_axis_param='p_value',
                    title="Approximation Ratio Medio vs. Layer QAOA (p)"
                )
            elif choice == "4":
                # Confronto degli ottimizzatori
                # Raggruppiamo per ottimizzatore ed estraiamo l'approximation ratio medio
                opt_data = collections.defaultdict(list)
                for entry in qaoa_results:
                    config = entry.get('qaoa_config', {})
                    opt = config.get('optimizer', config.get('optimizer_method', 'COBYLA'))
                    ratio = entry.get('metrics', {}).get('approximation_ratio', 0.0)
                    opt_data[opt].append(ratio)
                    
                plt.figure(figsize=(10, 6))
                opts = list(opt_data.keys())
                means = [np.mean(opt_data[o]) for o in opts]
                stds = [np.std(opt_data[o]) for o in opts]
                
                bars = plt.bar(opts, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=['#3498db', '#2ecc71', '#e74c3c'])
                plt.ylabel('Approximation Ratio Medio')
                plt.title('Confronto delle Performance degli Ottimizzatori Classici')
                plt.ylim(0, 1.1)
                
                # Aggiungi i valori sulle barre
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
                    
                plt.grid(axis='y', linestyle=':', alpha=0.6)
                plt.show()
