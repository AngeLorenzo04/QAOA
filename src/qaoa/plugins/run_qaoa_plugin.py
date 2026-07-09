import sys
import os
import networkx as nx
import numpy as np

from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
from src.common.plotting import plot_qaoa_dashboard

class RunQAOAPlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "run_qaoa"

    @property
    def description(self) -> str:
        return "Esegui ottimizzazione QAOA standard e visualizza la dashboard"

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        # 1. Seleziona Ottimizzatore
        optimizer = Prompt.ask("Scegli l'ottimizzatore classico", choices=['COBYLA', 'SLSQP', 'GD'], default='COBYLA')
        shots = IntPrompt.ask("Scegli il numero di shots", default=1024)
        
        # 2. Calcola MaxCut classico se non disponibile
        exact_maxcut_val = graph.graph.get('exact_max_cut_value', -1)
        if exact_maxcut_val == -1:
            with console.status("[bold yellow]Calcolo del Max-Cut esatto classica...[/bold yellow]"):
                if n_nodes <= 16:
                    exact_res = find_exact_maxcut(graph)
                else:
                    exact_res = find_exact_maxcut_ilp(graph)
                exact_maxcut_val = exact_res['max_cut_value']
                graph.graph['exact_max_cut_value'] = exact_maxcut_val

        # 3. Esegui QAOA
        with console.status(f"[bold magenta]Esecuzione di QAOA con {optimizer}...[/bold magenta]"):
            qaoa_results = runner.run(
                optimizer_method=optimizer,
                shots=shots,
                max_optimization_iterations=100
            )

        # 4. Mostra i risultati
        best_bitstring = qaoa_results['best_measured_bitstring']
        expected_cut = qaoa_results['qaoa_expected_cut_value']
        best_cut = qaoa_results['best_measured_cut_value']
        iterations = qaoa_results['metrics']['optimization_iterations']
        termination_reason = qaoa_results['metrics']['termination_reason']
        
        approx_ratio_expected = expected_cut / exact_maxcut_val if exact_maxcut_val > 0 else 0.0
        approx_ratio_best = best_cut / exact_maxcut_val if exact_maxcut_val > 0 else 0.0

        result_text = (
            f"Taglio Massimo Esatto (Classico): {exact_maxcut_val}\n"
            f"Taglio QAOA Atteso: {expected_cut:.4f} (Ratio: {approx_ratio_expected:.4f})\n"
            f"Miglior Taglio Misurato: {best_cut} (Ratio: {approx_ratio_best:.4f})\n"
            f"Migliore Stringa di Bit: {best_bitstring}\n"
            f"Iterazioni Ottimizzatore: {iterations}\n"
            f"Motivo Termine: {termination_reason}"
        )
        console.print(Panel(result_text, title="[bold green]Risultati Ottimizzazione[/bold green]", expand=False))

        # 5. Visualizzazione Dashboard
        plot_choice = Prompt.ask("Vuoi visualizzare la dashboard grafica dei risultati?", choices=["si", "no"], default="si")
        if plot_choice.lower() in ["si", "s"]:
            # Costruisci l'array delle probabilità
            num_qubits = graph.number_of_nodes()
            probs = np.zeros(2**num_qubits)
            quasi_dist = qaoa_results['quasi_distribution']
            for bitstring, prob in quasi_dist.items():
                state_int = int(bitstring, 2)
                probs[state_int] = prob
                
            plot_qaoa_dashboard(
                graph=graph,
                k=2,
                probs=probs,
                best_bitstring=best_bitstring,
                cost_history=qaoa_results['metrics']['optimization_history'],
                title=f"QAOA N={n_nodes}, D={graph_info['density_edges']:.2f}, ID={graph_info['id']} ({optimizer})"
            )
