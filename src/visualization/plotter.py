import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import numpy as np
import collections

def plot_graph_with_cut(
    graph: nx.Graph, 
    partition: List[int], 
    title: str = "Graph with Max Cut",
    filepath: Optional[str] = None
) -> None:
    """
    Plots a graph, coloring nodes based on a given partition and highlighting cut edges.

    Args:
        graph (nx.Graph): The input NetworkX graph.
        partition (List[int]): A list where partition[i] = 0 or 1, indicating
                                which set node i belongs to.
        title (str): Title of the plot.
        filepath (Optional[str]): If provided, saves the plot to this path.
                                  Otherwise, displays the plot.
    """
    if not graph.number_of_nodes():
        print("Graph has no nodes to plot.")
        return

    pos = nx.spring_layout(graph, seed=42) # For consistent layout

    # Color nodes based on partition
    node_colors = ['skyblue' if partition[node] == 0 else 'salmon' for node in graph.nodes()]

    # Determine cut edges
    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges():
        # Ensure nodes are in the partition list (important if graph nodes are not 0 to N-1)
        # Assuming nodes are 0 to N-1 for simplicity, as per problem description
        if partition[u] != partition[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    plt.figure(figsize=(10, 8))
    
    # Draw non-cut edges first (thinner, gray)
    nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='lightgray', width=1)
    
    # Draw cut edges (thicker, black)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='black', width=2)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    plt.title(title)
    plt.axis('off') # Hide axes

    if filepath:
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to {filepath}")
    else:
        plt.show()
        plt.close()

def plot_optimizer_convergence(
    optimization_history: Dict[str, List[Any]],
    title: str = "Optimizer Convergence",
    filepath: Optional[str] = None
) -> None:
    """
    Plots the convergence of the classical optimizer during QAOA parameter optimization.

    Args:
        optimization_history (Dict[str, List[Any]]): The history dictionary from qaoa_optimizer,
                                                    containing 'fun' (objective values)
                                                    and 'params' (parameter values).
        title (str): Title of the plot.
        filepath (Optional[str]): If provided, saves the plot to this path.
                                  Otherwise, displays the plot.
    """
    if not optimization_history or not optimization_history.get('fun'):
        print("No optimization history data to plot.")
        return

    objective_values = optimization_history['fun']
    iterations = range(len(objective_values))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objective_values, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Objective Function Value (Negative MaxCut)")
    plt.grid(True)

    if filepath:
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to {filepath}")
    else:
        plt.show()
        plt.close()

def plot_probability_distribution(
    quasi_distribution: Dict[str, float],
    title: str = "Probability Distribution of Bitstrings",
    optimal_bitstring: Optional[str] = None,
    filepath: Optional[str] = None,
    top_n: int = 10 # Display top N bitstrings for clarity
) -> None:
    """
    Plots the probability distribution of bitstrings from QAOA results.

    Args:
        quasi_distribution (Dict[str, float]): A dictionary where keys are bitstrings
                                                and values are their probabilities.
        title (str): Title of the plot.
        optimal_bitstring (Optional[str]): The bitstring representing the known optimal solution,
                                            to be highlighted.
        filepath (Optional[str]): If provided, saves the plot to this path.
                                  Otherwise, displays the plot.
        top_n (int): Number of top bitstrings to display in the plot.
    """
    if not quasi_distribution:
        print("No quasi-distribution data to plot.")
        return

    # Sort bitstrings by probability in descending order
    sorted_dist = sorted(quasi_distribution.items(), key=lambda item: item[1], reverse=True)

    # Take top N for plotting if there are too many
    if len(sorted_dist) > top_n:
        sorted_dist = sorted_dist[:top_n]
        title += f" (Top {top_n})"

    bitstrings = [item[0] for item in sorted_dist]
    probabilities = [item[1] for item in sorted_dist]
    
    # Color bars: highlight optimal bitstring if present
    bar_colors = ['skyblue' if bs != optimal_bitstring else 'salmon' for bs in bitstrings]

    plt.figure(figsize=(12, 7))
    plt.bar(bitstrings, probabilities, color=bar_colors)
    plt.xlabel("Bitstring")
    plt.ylabel("Probability")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    if filepath:
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to {filepath}")
    else:
        plt.show()
        plt.close()

def plot_approximation_ratio_vs_params(
    qaoa_results_summary: List[Dict[str, Any]],
    x_axis_param: str, # 'n_vertices', 'density_edges', or 'p_value'
    title: str = "Approximation Ratio vs. Parameter",
    filepath: Optional[str] = None
) -> None:
    """
    Plots the approximation ratio against a specified QAOA parameter (N, D, or p).

    Args:
        qaoa_results_summary (List[Dict[str, Any]]): A list of dictionaries,
                                                    each representing a QAOA run result
                                                    from qaoa_benchmarking_summary.json.
        x_axis_param (str): The parameter to plot on the x-axis.
                            Can be 'n_vertices', 'density_edges', or 'p_value'.
        title (str): Title of the plot.
        filepath (Optional[str]): If provided, saves the plot to this path.
                                  Otherwise, displays the plot.
    """
    if not qaoa_results_summary:
        print("No QAOA results to plot approximation ratio.")
        return

    valid_x_params = ['n_vertices', 'density_edges', 'p_value']
    if x_axis_param not in valid_x_params:
        print(f"Error: x_axis_param must be one of {valid_x_params}. Got '{x_axis_param}'.")
        return

    # Group data by the x_axis_param and calculate average approximation ratio
    grouped_data = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for entry in qaoa_results_summary:
        n_vertices = entry['graph_metadata']['n_vertices']
        density_edges = entry['graph_metadata']['density_edges']
        p_value = entry['qaoa_config']['p_value']
        approx_ratio = entry['metrics']['approximation_ratio']

        # Determine the key for grouping based on x_axis_param
        if x_axis_param == 'n_vertices':
            key = n_vertices
            grouped_data[key]['ratios'].append(approx_ratio)
            # Group by p_value and density for different lines if plotting N
            grouped_data[key]['p'].append(p_value)
            grouped_data[key]['density'].append(density_edges)
        elif x_axis_param == 'density_edges':
            key = density_edges
            grouped_data[key]['ratios'].append(approx_ratio)
            # Group by p_value and N for different lines if plotting D
            grouped_data[key]['p'].append(p_value)
            grouped_data[key]['n_vertices'].append(n_vertices)
        elif x_axis_param == 'p_value':
            key = p_value
            grouped_data[key]['ratios'].append(approx_ratio)
            # Group by N and density for different lines if plotting p
            grouped_data[key]['n_vertices'].append(n_vertices)
            grouped_data[key]['density'].append(density_edges)

    # Prepare data for plotting
    x_values = sorted(list(grouped_data.keys()))
    
    plt.figure(figsize=(12, 7))

    if x_axis_param == 'n_vertices' or x_axis_param == 'p_value':
        # For N or p, we might want to see different lines for different densities or N
        # Let's plot average ratios for unique combinations of other parameters
        plot_lines = collections.defaultdict(list) # Key: (other_param1, other_param2)
        
        for x_val in x_values:
            current_ratios = grouped_data[x_val]['ratios']
            current_ps = grouped_data[x_val]['p']
            current_densities = grouped_data[x_val]['density']
            current_ns = grouped_data[x_val]['n_vertices']

            if x_axis_param == 'n_vertices':
                # Group by density and p_value
                sub_grouped = collections.defaultdict(list)
                for i in range(len(current_ratios)):
                    sub_grouped[(current_densities[i], current_ps[i])].append(current_ratios[i])
                for (density, p_val), ratios in sub_grouped.items():
                    plot_lines[(density, p_val)].append((x_val, np.mean(ratios)))
            elif x_axis_param == 'p_value':
                # Group by n_vertices and density
                sub_grouped = collections.defaultdict(list)
                for i in range(len(current_ratios)):
                    sub_grouped[(current_ns[i], current_densities[i])].append(current_ratios[i])
                for (n_val, density), ratios in sub_grouped.items():
                    plot_lines[(n_val, density)].append((x_val, np.mean(ratios)))
            
        # Plot each line
        for key_tuple, data_points in plot_lines.items():
            sorted_data = sorted(data_points)
            x = [item[0] for item in sorted_data]
            y = [item[1] for item in sorted_data]
            label = ""
            if x_axis_param == 'n_vertices':
                label = f"D={key_tuple[0]:.2f}, p={key_tuple[1]}"
            elif x_axis_param == 'p_value':
                label = f"N={key_tuple[0]}, D={key_tuple[1]:.2f}"
            plt.plot(x, y, marker='o', linestyle='-', label=label)
        plt.legend(title=f"Fixed Parameters")

    elif x_axis_param == 'density_edges':
        plot_lines = collections.defaultdict(list) # Key: (n_vertices, p_value)

        for x_val in x_values:
            current_ratios = grouped_data[x_val]['ratios']
            current_ns = grouped_data[x_val]['n_vertices']
            current_ps = grouped_data[x_val]['p']

            # Group by n_vertices and p_value
            sub_grouped = collections.defaultdict(list)
            for i in range(len(current_ratios)):
                sub_grouped[(current_ns[i], current_ps[i])].append(current_ratios[i])
            for (n_val, p_val), ratios in sub_grouped.items():
                plot_lines[(n_val, p_val)].append((x_val, np.mean(ratios)))
        
        # Plot each line
        for key_tuple, data_points in plot_lines.items():
            sorted_data = sorted(data_points)
            x = [item[0] for item in sorted_data]
            y = [item[1] for item in sorted_data]
            label = f"N={key_tuple[0]}, p={key_tuple[1]}"
            plt.plot(x, y, marker='o', linestyle='-', label=label)
        plt.legend(title=f"Fixed Parameters")

    plt.title(title)
    plt.xlabel(x_axis_param.replace('_', ' ').title())
    plt.ylabel("Average Approximation Ratio")
    plt.grid(True)

    if filepath:
        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to {filepath}")
    else:
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Example Usage:
    # 1. Simple graph with a known max cut
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3)])
    
    # Max cut for this graph could be {[0,3], [1,2]} with cut value 3
    # Partition: node 0 in set 0, node 1 in set 1, node 2 in set 1, node 3 in set 0
    max_cut_partition_example = [0, 1, 1, 0] 
    plot_graph_with_cut(G, max_cut_partition_example, "Example Graph with Max Cut (N=4)")

    # 2. A larger graph (e.g., a cycle graph)
    G_cycle = nx.cycle_graph(6)
    # For an even cycle graph, max cut is N/2 edges. 
    # Example partition for N=6: [0,1,0,1,0,1]
    max_cut_partition_cycle = [0, 1, 0, 1, 0, 1]
    plot_graph_with_cut(G_cycle, max_cut_partition_cycle, "Cycle Graph N=6 with Max Cut")

    # 3. Saving to a file
    G_path = nx.path_graph(5)
    max_cut_partition_path = [0, 1, 0, 1, 0] # Example for path graph
    plot_graph_with_cut(G_path, max_cut_partition_path, "Path Graph N=5 with Max Cut (Saved)", filepath="path_graph_max_cut.png")

    # Example for optimizer convergence plot
    dummy_history = {
        'fun': [-5, -4.5, -4.8, -3.9, -3.5, -3.6, -3.4, -3.3, -3.2, -3.1, -3.0],
        'params': [[0.1, 0.2], [0.15, 0.25], [0.12, 0.22], [0.18, 0.28], [0.2, 0.3],
                   [0.21, 0.31], [0.22, 0.32], [0.23, 0.33], [0.24, 0.34], [0.25, 0.35], [0.26, 0.36]]
    }
    plot_optimizer_convergence(dummy_history, "Dummy Optimizer Convergence Plot")
    plot_optimizer_convergence(dummy_history, "Dummy Optimizer Convergence Plot (Saved)", filepath="optimizer_convergence.png")

    # Example for probability distribution plot
    dummy_quasi_distribution = {
        '000': 0.1, '001': 0.05, '010': 0.15, '011': 0.08,
        '100': 0.2, '101': 0.12, '110': 0.18, '111': 0.12
    }
    plot_probability_distribution(dummy_quasi_distribution, "Dummy Probability Distribution")
    plot_probability_distribution(dummy_quasi_distribution, "Dummy Probability Distribution (Saved)", filepath="probability_distribution.png", optimal_bitstring='100')

    # Example for approximation ratio vs. parameter plot
    dummy_qaoa_results = [
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.5, 'seed': 1, 'id': 0}, 'qaoa_config': {'p_value': 1, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.8}},
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.5, 'seed': 2, 'id': 1}, 'qaoa_config': {'p_value': 1, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.75}},
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.5, 'seed': 3, 'id': 2}, 'qaoa_config': {'p_value': 2, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.9}},
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.5, 'seed': 4, 'id': 3}, 'qaoa_config': {'p_value': 2, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.88}},
        {'graph_metadata': {'n_vertices': 8, 'density_edges': 0.5, 'seed': 5, 'id': 4}, 'qaoa_config': {'p_value': 1, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.6}},
        {'graph_metadata': {'n_vertices': 8, 'density_edges': 0.5, 'seed': 6, 'id': 5}, 'qaoa_config': {'p_value': 1, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.55}},
        {'graph_metadata': {'n_vertices': 8, 'density_edges': 0.5, 'seed': 7, 'id': 6}, 'qaoa_config': {'p_value': 2, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.75}},
        {'graph_metadata': {'n_vertices': 8, 'density_edges': 0.5, 'seed': 8, 'id': 7}, 'qaoa_config': {'p_value': 2, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.72}},
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.25, 'seed': 9, 'id': 8}, 'qaoa_config': {'p_value': 1, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.85}},
        {'graph_metadata': {'n_vertices': 4, 'density_edges': 0.25, 'seed': 10, 'id': 9}, 'qaoa_config': {'p_value': 2, 'mixer': 'standard', 'encoding': 'binary'}, 'metrics': {'approximation_ratio': 0.92}},
    ]
    plot_approximation_ratio_vs_params(dummy_qaoa_results, 'n_vertices', "Approximation Ratio vs. Number of Vertices")
    plot_approximation_ratio_vs_params(dummy_qaoa_results, 'p_value', "Approximation Ratio vs. p-value", filepath="approx_ratio_vs_p.png")
    plot_approximation_ratio_vs_params(dummy_qaoa_results, 'density_edges', "Approximation Ratio vs. Edge Density", filepath="approx_ratio_vs_density.png")