import networkx as nx
import os
import pickle
import random
from typing import List, Dict, Any, Optional

from src.common.graphs import create_random_graph
from src.data.exact_maxcut_solver import find_exact_maxcut # Import the solver

def generate_and_save_graphs(
    n_vertices_list: List[int],
    density_edges_list: List[float],
    num_graphs_per_combo: int,
    output_dir: str
) -> None:
    """
    Generates a dataset of random graphs and saves them to disk.

    Args:
        n_vertices_list (List[int]): List of number of vertices to use.
        density_edges_list (List[float]): List of edge densities (probabilities) to use.
        num_graphs_per_combo (int): Number of unique graphs to generate for each
                                    (n_vertices, density) combination, using different seeds.
        output_dir (str): Directory to save the generated graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating graphs and saving to: {output_dir}")

    for n_vertices in n_vertices_list:
        for density in density_edges_list:
            for i in range(num_graphs_per_combo):
                seed = random.randint(0, 1000000) # Use a large range for seeds
                graph = create_random_graph(n_nodes=n_vertices, probability=density, seed=seed)
                
                # Add metadata to the graph for easy identification
                graph.graph['n_vertices'] = n_vertices
                graph.graph['density_edges'] = density
                graph.graph['seed'] = seed
                graph.graph['id'] = i # Unique ID for this combination

                # Calculate and store exact MaxCut for benchmarking
                exact_maxcut_result = find_exact_maxcut(graph)
                graph.graph['exact_max_cut_value'] = exact_maxcut_result['max_cut_value']
                # Storing all partitions can be memory intensive, consider storing only one or a summary if needed
                graph.graph['exact_max_cut_partitions'] = exact_maxcut_result['max_cut_partitions']

                filename = f"graph_n{n_vertices}_d{density:.2f}_id{i}_seed{seed}.gpickle"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(graph, f)
                print(f"  Saved {filename}")

def load_graphs(
    input_dir: str,
    n_vertex: Optional[int] = None,
    density: Optional[float] = None,
    graph_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Loads graphs from a specified directory, optionally filtering by properties.

    Args:
        input_dir (str): Directory containing the graph files.
        n_vertex (Optional[int]): Filter by number of vertices.
        density (Optional[float]): Filter by edge density.
        graph_id (Optional[int]): Filter by graph ID (index within combinations).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the graph
                              and its metadata.
    """
    loaded_graphs = []
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return []

    for filename in os.listdir(input_dir):
        if filename.endswith(".gpickle"):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    graph = pickle.load(f)
                
                # Check metadata for filtering
                match = True
                if n_vertex is not None and graph.graph.get('n_vertices') != n_vertex:
                    match = False
                if density is not None and abs(graph.graph.get('density_edges', -1) - density) > 1e-6: # Float comparison
                    match = False
                if graph_id is not None and graph.graph.get('id') != graph_id:
                    match = False
                
                if match:
                    loaded_graphs.append({
                        'graph': graph,
                        'n_vertices': graph.graph.get('n_vertices'),
                        'density_edges': graph.graph.get('density_edges'),
                        'seed': graph.graph.get('seed'),
                        'id': graph.graph.get('id')
                    })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return loaded_graphs

if __name__ == "__main__":
    # Example usage:
    N_VERTICES = [4, 8] # Reduced for quick test
    DENSITY_EDGES = [0.1, 0.5] # Reduced for quick test
    NUM_GRAPHS_PER_COMBO = 2 # Reduced for quick test
    OUTPUT_DIR = "generated_graphs_test"

    generate_and_save_graphs(N_VERTICES, DENSITY_EDGES, NUM_GRAPHS_PER_COMBO, OUTPUT_DIR)

    print("\nLoading all generated graphs:")
    all_graphs = load_graphs(OUTPUT_DIR)
    print(f"Loaded {len(all_graphs)} graphs.")

    print("\nLoading specific graphs (n_vertex=4, density=0.1):")
    filtered_graphs = load_graphs(OUTPUT_DIR, n_vertex=4, density=0.1)
    print(f"Loaded {len(filtered_graphs)} filtered graphs.")
    if filtered_graphs:
        print(f"First filtered graph: N={filtered_graphs[0]['n_vertices']}, D={filtered_graphs[0]['density_edges']:.2f}, ID={filtered_graphs[0]['id']}")
