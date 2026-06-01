import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(graph: nx.Graph) -> None:
    """
    Visualize the graph used for the Max-Cut problem.
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
    Visualize the Max-Cut result by separating nodes into two distinct columns.
    This ensures no overlap between the two partitions and clearly shows the boundary.
    
    Args:
        graph: The problem graph.
        bitstring: The best solution found (e.g., '0101').
    """
    # 1. Identify partitions based on the bitstring
    group_a = [i for i, bit in enumerate(bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(bitstring) if bit == '1']
    
    # 2. Create a custom bipartite-like layout to guarantee separation
    # Group A on the left (x=-1), Group B on the right (x=1)
    pos = {}
    for i, node in enumerate(group_a):
        pos[node] = np.array([-1, i - (len(group_a)-1)/2.0])
    for i, node in enumerate(group_b):
        pos[node] = np.array([1, i - (len(group_b)-1)/2.0])

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 3. Draw non-overlapping background panels
    # Left panel (Group A)
    ax.add_patch(plt.Rectangle((-1.5, -max(len(group_a), len(group_b))), 1.4, 2*max(len(group_a), len(group_b)), 
                               color='blue', alpha=0.1, label='Partition A (0)'))
    # Right panel (Group B)
    ax.add_patch(plt.Rectangle((0.1, -max(len(group_a), len(group_b))), 1.4, 2*max(len(group_a), len(group_b)), 
                               color='red', alpha=0.1, label='Partition B (1)'))

    # 4. Draw edges
    cut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] != bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] == bitstring[v]]

    # Cut edges represent the "Boundary" crossing the middle
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, width=3, 
                           edge_color='black', style='-', alpha=0.8)
    
    # Uncut edges remain inside their respective partitions
    nx.draw_networkx_edges(graph, pos, edgelist=uncut_edges, width=1.5, 
                           edge_color='gray', style='--', alpha=0.4)

    # 5. Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=group_a, node_color='blue', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos, nodelist=group_b, node_color='red', node_size=800, edgecolors='black')
    
    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=14, font_color='white', font_weight='bold')

    plt.title(f"Max-Cut Result: Bitstring {bitstring}\nNodes partitioned into two non-overlapping areas", fontsize=15)
    
    # Custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=0, marker='o', markersize=10, label='Group A (0)'),
        Line2D([0], [0], color='red', lw=0, marker='o', markersize=10, label='Group B (1)'),
        Line2D([0], [0], color='black', lw=2, label='Cut Edge (Boundary)'),
        Line2D([0], [0], color='gray', lw=1, ls='--', label='Uncut Edge')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
