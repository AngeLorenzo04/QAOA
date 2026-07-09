import os
import json
import networkx as nx
from typing import List
from tqdm import tqdm

from rich.prompt import Prompt, IntPrompt

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.data.graph_dataset_generator import generate_and_save_graphs, load_graphs
from src.data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp

# Configuration
N_VERTICES = [4, 8, 16]
DENSITY_EDGES = [0.1, 0.25, 0.5, 0.75]
NUM_GRAPHS_PER_COMBO = 10
GRAPH_OUTPUT_DIR = "data/generated_graphs"
BENCHMARK_RESULTS_DIR = "data/benchmarking_results"

class BenchmarkingPlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "benchmarking"

    @property
    def description(self) -> str:
        return "Esegui il benchmarking completo (classico ILP e/o quantistico QAOA) sul dataset"

    @property
    def requires_graph(self) -> bool:
        return False

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
        os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
        
        console.print("\n[bold magenta]=== CONFIGURAZIONE BENCHMARK ===[/bold magenta]")
        
        # 1. Scegli tipo calcolo
        console.print("Scegli la modalità di calcolo:")
        console.print("  [bold green]c[/bold green]: Solo risposta classica (Generazione grafi e calcolo Max-Cut ILP)")
        console.print("  [bold cyan]q[/bold cyan]: Solo risposta quantistica (Simulazione QAOA)")
        console.print("  [bold yellow]b[/bold yellow]: Entrambe (Classica + Quantistica)")
        mode = Prompt.ask("Seleziona modalità", choices=["c", "q", "b"], default="b")
        
        # 2. Seleziona ottimizzatore per la parte quantistica
        active_optimizers = ["COBYLA"]
        p_layers_to_run = [1, 2, 3]
        
        if mode in ["q", "b"]:
            opt_choice = Prompt.ask(
                "Scegli gli ottimizzatori classici da testare (separati da spazio o 'ALL')",
                default="COBYLA"
            ).strip()
            if opt_choice.upper() == "ALL":
                active_optimizers = ["COBYLA", "SLSQP", "GD"]
            else:
                active_optimizers = [o.strip() for o in opt_choice.split() if o.strip() in ["COBYLA", "SLSQP", "GD"]]
                if not active_optimizers:
                    active_optimizers = ["COBYLA"]
                    
            p_choice = Prompt.ask(
                "Scegli i layer QAOA (p) da testare (separati da spazio o 'ALL')",
                default="1"
            ).strip()
            if p_choice.upper() == "ALL":
                p_layers_to_run = [1, 2, 3]
            else:
                p_layers_to_run = [int(p) for p in p_choice.split() if p.strip().isdigit()]
                if not p_layers_to_run:
                    p_layers_to_run = [1]
        
        # --- Fase Classica ---
        if mode in ["c", "b"]:
            console.print("\n[bold green]--- Avvio Fase Classica (Generazione e ILP) ---[/bold green]")
            # 1. Generate/Ensure Graphs exist
            console.print("Verifica/Generazione del dataset di grafi...")
            generate_and_save_graphs(N_VERTICES, DENSITY_EDGES, NUM_GRAPHS_PER_COMBO, GRAPH_OUTPUT_DIR)
            
            # 2. Load Graphs
            all_generated_graphs_info = load_graphs(GRAPH_OUTPUT_DIR)
            console.print(f"Dataset caricato: {len(all_generated_graphs_info)} grafi.")
            
            # 3. Solutore Classico ILP
            console.print("[bold green]Calcolo Max-Cut esatto classica (ILP)...[/bold green]")
            pbar_ilp = tqdm(all_generated_graphs_info, desc="Classico ILP")
            for g_info in pbar_ilp:
                graph: nx.Graph = g_info['graph']
                n_nodes = graph.number_of_nodes()
                pbar_ilp.set_postfix_str(f"N={g_info['n_vertices']} D={g_info['density_edges']:.2f} ID={g_info['id']}")
                
                if 'exact_max_cut_value' in graph.graph and graph.graph['exact_max_cut_value'] != -1:
                    continue
                    
                maxcut_results = find_exact_maxcut_ilp(graph)
                
                result_filepath = os.path.join(
                    BENCHMARK_RESULTS_DIR,
                    f"maxcut_ilp_n{g_info['n_vertices']}_d{g_info['density_edges']:.2f}_id{g_info['id']}.json"
                )
                
                full_result = {
                    'graph_metadata': {
                        'n_vertices': g_info['n_vertices'],
                        'density_edges': g_info['density_edges'],
                        'seed': g_info['seed'],
                        'id': g_info['id'],
                        'graph_filepath': os.path.join(GRAPH_OUTPUT_DIR, f"graph_n{g_info['n_vertices']}_d{g_info['density_edges']:.2f}_id{g_info['id']}.gpickle")
                    },
                    'exact_maxcut': maxcut_results
                }
                
                with open(result_filepath, 'w') as f:
                    json.dump(full_result, f, indent=4)
                    
            console.print("[bold green]Fase Classica Completata.[/bold green]")
            
        # --- Fase Quantistica ---
        if mode in ["q", "b"]:
            console.print("\n[bold cyan]--- Avvio Fase Quantistica (QAOA) ---[/bold cyan]")
            
            # Load graphs
            all_graphs_info = load_graphs(GRAPH_OUTPUT_DIR)
            valid_graphs_info = []
            
            for g_info in all_graphs_info:
                graph: nx.Graph = g_info['graph']
                n_nodes = graph.number_of_nodes()
                
                # Statevector simulation safety filter
                if n_nodes > 20:
                    continue
                    
                exact_max_cut_value = graph.graph.get('exact_max_cut_value', -1)
                
                # Check for ILP files
                if exact_max_cut_value == -1:
                    ilp_result_filepath = os.path.join(
                        BENCHMARK_RESULTS_DIR,
                        f"maxcut_ilp_n{g_info['n_vertices']}_d{g_info['density_edges']:.2f}_id{g_info['id']}.json"
                    )
                    if os.path.exists(ilp_result_filepath):
                        with open(ilp_result_filepath, 'r') as f:
                            ilp_data = json.load(f)
                            graph.graph['exact_max_cut_value'] = ilp_data['exact_maxcut']['max_cut_value']
                        valid_graphs_info.append(g_info)
                else:
                    valid_graphs_info.append(g_info)
                    
            total_runs = len(valid_graphs_info) * len(p_layers_to_run) * len(active_optimizers)
            qaoa_results = []
            
            if total_runs == 0:
                console.print("[yellow]Nessun grafo valido per QAOA (N <= 20 con Max-Cut disponibile) trovato.[/yellow]")
                return
                
            console.print(f"Esecuzione di {total_runs} simulazioni QAOA...")
            
            with tqdm(total=total_runs, desc="QAOA Benchmarking") as pbar:
                for g_info in valid_graphs_info:
                    graph: nx.Graph = g_info['graph']
                    n_nodes = graph.number_of_nodes()
                    exact_max_cut_value = graph.graph['exact_max_cut_value']
                    
                    for p_val in p_layers_to_run:
                        for optimizer_method in active_optimizers:
                            pbar.set_postfix_str(f"N={n_nodes} D={g_info['density_edges']:.2f} ID={g_info['id']} Opt={optimizer_method} p={p_val}")
                            
                            runner = QAOARunner(
                                graph=graph,
                                p_value=p_val,
                                mixer_type="standard",
                                encoding_type="binary"
                            )
                            
                            # Run
                            qaoa_run_results = runner.run(
                                max_optimization_iterations=100,
                                optimizer_method=optimizer_method,
                                shots=1024,
                                epsilon=1e-5,
                                timeout=180.0
                            )
                            
                            qaoa_cut_value = qaoa_run_results['best_measured_cut_value']
                            approximation_ratio = qaoa_cut_value / exact_max_cut_value if exact_max_cut_value > 0 else 0.0
                            
                            qaoa_result_entry = {
                                'graph_metadata': {
                                    'n_vertices': g_info['n_vertices'],
                                    'density_edges': g_info['density_edges'],
                                    'seed': g_info['seed'],
                                    'id': g_info['id'],
                                    'filepath': os.path.join(GRAPH_OUTPUT_DIR, f"graph_n{g_info['n_vertices']}_d{g_info['density_edges']:.2f}_id{g_info['id']}.gpickle")
                                },
                                'exact_max_cut_value': exact_max_cut_value,
                                'qaoa_config': {
                                    'p_value': p_val,
                                    'mixer': "standard",
                                    'encoding': "binary",
                                    'optimizer': optimizer_method
                                },
                                'qaoa_results': {
                                    'optimal_params': qaoa_run_results['optimal_params'],
                                    'qaoa_expected_cut_value': qaoa_run_results['qaoa_expected_cut_value'],
                                    'best_measured_cut_value': qaoa_run_results['best_measured_cut_value'],
                                    'best_measured_bitstring': qaoa_run_results['best_measured_bitstring'],
                                    'quasi_distribution': qaoa_run_results['quasi_distribution'],
                                },
                                'metrics': {
                                    'qaoa_cut_value': qaoa_cut_value,
                                    'approximation_ratio': approximation_ratio,
                                    'circuit_depth': qaoa_run_results['metrics']['circuit_depth'],
                                    'num_parameters': qaoa_run_results['metrics']['num_parameters'],
                                    'optimization_iterations': qaoa_run_results['metrics']['optimization_iterations'],
                                    'termination_reason': qaoa_run_results['metrics'].get('termination_reason', 'optimizer_completed'),
                                    'optimization_history': qaoa_run_results['metrics']['optimization_history'],
                                    'total_shots': qaoa_run_results['metrics']['total_shots']
                                }
                            }
                            qaoa_results.append(qaoa_result_entry)
                            pbar.update(1)
                            
            # Save summary
            qaoa_output_filepath = os.path.join(BENCHMARK_RESULTS_DIR, "qaoa_benchmarking_summary.json")
            with open(qaoa_output_filepath, 'w') as f:
                json.dump(qaoa_results, f, indent=4, sort_keys=True)
                
            console.print(f"[bold cyan]Fase Quantistica Completata. Sommario salvato in: {qaoa_output_filepath}[/bold cyan]")
