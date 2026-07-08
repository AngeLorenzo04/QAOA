import os
import json
import argparse
import networkx as nx
from typing import Dict, Any, List # Added List to imports

from src.data.graph_dataset_generator import generate_and_save_graphs, load_graphs
from src.data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
from src.qaoa.qaoa_runner import QAOARunner # Import QAOARunner # Import the ansatz function

# --- Configuration for Graph Generation ---
N_VERTICES = [4, 8, 16, 32]
DENSITY_EDGES = [0.1, 0.25, 0.5, 0.75]
NUM_GRAPHS_PER_COMBO = 10
GRAPH_OUTPUT_DIR = "data/generated_graphs"
BENCHMARK_RESULTS_DIR = "data/benchmarking_results"

# --- Configuration for QAOA Benchmarking ---
P_VALUES: List[int] = [1, 2, 3] # Number of QAOA layers
MIXERS: List[str] = ["standard"] # Currently only "standard" is implemented
ENCODINGS: List[str] = ["binary"] # Currently only "binary" is implemented
OPTIMIZERS: List[str] = ["COBYLA"] # Available options: "COBYLA", "SLSQP", "GD"

def execute_benchmarking_setup():
    """
    Executes the first phase of the benchmarking setup:
    1. Generates a dataset of graphs.
    2. For each graph, finds its exact MaxCut value.
    3. Saves the graph metadata and MaxCut results.
    """
    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
    print("--- Starting Benchmarking Setup ---")

    # 1. Generate Graphs
    print("\nGenerating and saving graphs...")
    generate_and_save_graphs(N_VERTICES, DENSITY_EDGES, NUM_GRAPHS_PER_COMBO, GRAPH_OUTPUT_DIR)
    print("Graph generation complete.")

    # 2. Load Graphs and Determine MaxCut (either from pre-calculated or ILP)
    print("\nLoading graphs and preparing MaxCut results...")
    all_generated_graphs_info = load_graphs(GRAPH_OUTPUT_DIR)
    print(f"Loaded {len(all_generated_graphs_info)} graphs.")

    for graph_info in all_generated_graphs_info:
        graph: nx.Graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()

        # Check if MaxCut was already calculated and stored during generation
        if 'exact_max_cut_value' in graph.graph and graph.graph['exact_max_cut_value'] != -1:
            print(f"  MaxCut for N={n_nodes}, D={graph_info['density_edges']:.2f}, ID={graph_info['id']} "
                  f"already pre-calculated (Value: {graph.graph['exact_max_cut_value']}).")
            # No need to recalculate or save separate JSON if already in gpickle
            continue 
        
        # If not pre-calculated (likely N=32 or similar large graph), use ILP
        print(f"  Calculating MaxCut for N={n_nodes}, D={graph_info['density_edges']:.2f}, ID={graph_info['id']} using ILP...")

        maxcut_results = find_exact_maxcut_ilp(graph)
        
        # Update the graph object with ILP results before saving/re-saving (optional, depends on workflow)
        # For now, we'll save these to separate JSONs to distinguish ILP from brute-force
        # and not modify the already saved gpickle
        
        result_filepath = os.path.join(
            BENCHMARK_RESULTS_DIR,
            f"maxcut_ilp_n{graph_info['n_vertices']}_d{graph_info['density_edges']:.2f}_id{graph_info['id']}_seed{graph_info['seed']}.json"
        )
        
        full_result = {
            'graph_metadata': {
                'n_vertices': graph_info['n_vertices'],
                'density_edges': graph_info['density_edges'],
                'seed': graph_info['seed'],
                'id': graph_info['id'],
                'graph_filepath': os.path.join(GRAPH_OUTPUT_DIR, f"graph_n{graph_info['n_vertices']}_d{graph_info['density_edges']:.2f}_id{graph_info['id']}_seed{graph_info['seed']}.gpickle")
            },
            'exact_maxcut': maxcut_results
        }
        
        with open(result_filepath, 'w') as f:
            json.dump(full_result, f, indent=4)
        print(f"  Saved ILP MaxCut results to {os.path.basename(result_filepath)}")
        
    print("\n--- Benchmarking Setup Complete ---")

def run_qaoa_benchmarking(optimizers_list: List[str] = None):
    """
    Executes the QAOA benchmarking phase:
    1. Loads generated graphs and their exact MaxCut solutions.
    2. Runs QAOA for each graph with different configurations (mixers, encodings, p-layers).
    3. Collects and saves various QAOA performance metrics.
    """
    print("\n--- Starting QAOA Benchmarking ---")
    active_optimizers = optimizers_list if optimizers_list is not None else OPTIMIZERS
    
    # 1. Load Graphs and their Exact MaxCut Solutions
    all_graphs_info = load_graphs(GRAPH_OUTPUT_DIR)
    
    qaoa_results = []

    for graph_info in all_graphs_info:
        graph: nx.Graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()

        exact_max_cut_value = graph.graph.get('exact_max_cut_value')

        # If exact_max_cut_value was -1 (meaning brute-force skipped it),
        # try to load it from the ILP results JSON.
        if exact_max_cut_value == -1:
            ilp_result_filepath = os.path.join(
                BENCHMARK_RESULTS_DIR,
                f"maxcut_ilp_n{graph_info['n_vertices']}_d{graph_info['density_edges']:.2f}_id{graph_info['id']}_seed{graph_info['seed']}.json"
            )
            if os.path.exists(ilp_result_filepath):
                with open(ilp_result_filepath, 'r') as f:
                    ilp_data = json.load(f)
                    exact_max_cut_value = ilp_data['exact_maxcut']['max_cut_value']
            else:
                print(f"Warning: Exact MaxCut for graph N={n_nodes}, ID={graph_info['id']} not found. Skipping QAOA.")
                continue

        print(f"\nRunning QAOA for graph N={n_nodes}, D={graph_info['density_edges']:.2f}, ID={graph_info['id']} "
              f"(Exact MaxCut: {exact_max_cut_value})")

        if n_nodes > 20:
            print(f"Warning: Exact statevector simulation for {n_nodes} qubits is too memory-intensive for the default Qiskit reference Sampler (requires ~{2**(n_nodes)*16/1024**3:.1f} GB RAM). Skipping QAOA for this graph.")
            continue

        for p_val in P_VALUES:
            for mixer_type in MIXERS:
                for encoding_type in ENCODINGS:
                    for optimizer_method in active_optimizers:
                        print(f"  Config: p={p_val}, mixer={mixer_type}, encoding={encoding_type}, optimizer={optimizer_method}")
    
                        # Instantiate QAOA Runner
                        runner = QAOARunner(
                            graph=graph,
                            p_value=p_val,
                            mixer_type=mixer_type,
                            encoding_type=encoding_type
                        )
    
                        # Run QAOA and get results
                        qaoa_run_results = runner.run(
                            max_optimization_iterations=100,
                            optimizer_method=optimizer_method,
                            shots=1024,
                            epsilon=1e-5,
                            timeout=180.0
                        )
                        
                        # Calculate approximation ratio
                        qaoa_cut_value = qaoa_run_results['best_measured_cut_value']
                        approximation_ratio = qaoa_cut_value / exact_max_cut_value if exact_max_cut_value > 0 else 0
    
                        qaoa_result_entry = {
                            'graph_metadata': {
                                'n_vertices': graph_info['n_vertices'],
                                'density_edges': graph_info['density_edges'],
                                'seed': graph_info['seed'],
                                'id': graph_info['id'],
                            },
                            'exact_max_cut_value': exact_max_cut_value,
                            'qaoa_config': {
                                'p_value': p_val,
                                'mixer': mixer_type,
                                'encoding': encoding_type,
                                'optimizer': optimizer_method
                            },
                            'qaoa_results': {
                                'optimal_params': qaoa_run_results['optimal_params'],
                                'qaoa_expected_cut_value': qaoa_run_results['qaoa_expected_cut_value'],
                                'best_measured_cut_value': qaoa_run_results['best_measured_cut_value'],
                                'best_measured_bitstring': qaoa_run_results['best_measured_bitstring'],
                                # 'quasi_distribution': qaoa_run_results['quasi_distribution'] # Can be large, consider if needed
                            },
                            'metrics': {
                                'qaoa_cut_value': qaoa_cut_value,
                                'approximation_ratio': approximation_ratio,
                                'circuit_depth': qaoa_run_results['metrics']['circuit_depth'],
                                'num_parameters': qaoa_run_results['metrics']['num_parameters'],
                                'optimization_iterations': qaoa_run_results['metrics']['optimization_iterations'],
                                'termination_reason': qaoa_run_results['metrics'].get('termination_reason', 'optimizer_completed'),
                                'optimization_history': qaoa_run_results['metrics']['optimization_history'], # Add optimization history
                                'total_shots': qaoa_run_results['metrics']['total_shots']
                            }
                        }
                        qaoa_results.append(qaoa_result_entry)
                        print(f"  QAOA run complete. Best cut: {qaoa_cut_value}, Approx Ratio: {approximation_ratio:.4f}")
    
    # --- Save QAOA Benchmarking Results ---
    qaoa_output_filepath = os.path.join(BENCHMARK_RESULTS_DIR, "qaoa_benchmarking_summary.json")
    # Use json.dumps with sort_keys=True for consistent output (useful for diffs)
    with open(qaoa_output_filepath, 'w') as f:
        json.dump(qaoa_results, f, indent=4, sort_keys=True)
    print(f"\nSaved QAOA benchmarking summary to {qaoa_output_filepath}")

    print("\n--- QAOA Benchmarking Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QAOA Benchmarking Pipeline")
    parser.add_argument('--setup-only', action='store_true', help="Run only the mathematical calculation (graph generation and ILP exact MaxCut)")
    parser.add_argument('--qaoa-only', action='store_true', help="Run only the quantum calculation (QAOA benchmarking)")
    parser.add_argument('--optimizers', nargs='+', type=str, help="List of optimizers to use (e.g. COBYLA SLSQP GD). Overrides the default in the script.")
    args = parser.parse_args()

    if args.setup_only:
        execute_benchmarking_setup()
    elif args.qaoa_only:
        run_qaoa_benchmarking(args.optimizers)
    else:
        execute_benchmarking_setup()
        run_qaoa_benchmarking(args.optimizers)
