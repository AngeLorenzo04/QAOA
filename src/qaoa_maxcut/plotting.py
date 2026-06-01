import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(graph: nx.Graph) -> None:
    """
    Visualize the graph used for the Max-Cut problem.
    
    Args:
        graph: The NetworkX graph to plot.
    """
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    plt.title("Graph Visualization (Original)")
    plt.show()


def plot_probabilities(probs, n_wires: int) -> None:
    """
    Visualize the bitstring probabilities as a bar chart.
    
    Args:
        probs: Array of probabilities for each basis state.
        n_wires: Number of qubits (nodes in the graph).
    """
    bitstrings = [format(i, f'0{n_wires}b') for i in range(2**n_wires)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(bitstrings, probs, color='skyblue')
    plt.xlabel("Bitstrings (Basis States)")
    plt.ylabel("Probability")
    plt.title("Measurement Probabilities after Optimization")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_result_graph(graph: nx.Graph, bitstring: str) -> None:
    """
    Visualize the Max-Cut result by coloring nodes and highlighting cut edges.
    
    Args:
        graph: The problem graph.
        bitstring: The best solution found (e.g., '0101').
    """
    # Map bitstring to node colors: '0' -> Group A (blue), '1' -> Group B (red)
    color_map = ['skyblue' if bitstring[i] == '0' else 'salmon' for i in range(len(bitstring))]
    
    # Identify which edges are "cut" (nodes are in different groups)
    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges():
        if bitstring[u] != bitstring[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=700)
    nx.draw_networkx_labels(graph, pos, font_size=16, font_family='sans-serif')
    
    # Draw cut edges (dashed)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, width=3, edge_color='green', style='dashed', label='Cut edges')
    # Draw non-cut edges (solid)
    nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, width=1, edge_color='gray', label='Uncut edges')
    
    plt.title(f"Max-Cut Solution: {bitstring}\n(Blue: Group A, Red: Group B)")
    plt.legend()
    plt.axis('off')
    plt.show()
