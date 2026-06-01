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


def plot_cost_history(cost_history: list) -> None:
    """
    Visualize the cost (expectation value) during optimization.
    
    Args:
        cost_history: List of cost values at each step.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'o-', color='tab:blue', label='Cost (Expectation Value)')
    plt.xlabel("Optimization Steps")
    plt.ylabel("Cost")
    plt.title("QAOA Optimization Convergence")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_result_graph(graph: nx.Graph, bitstring: str) -> None:
    """
    Visualize the Max-Cut result with a clear, single Decision Boundary.
    Uses a bipartite layout to ensure all Group A nodes are on one side
    and Group B nodes are on the other, making the boundary unambiguous.
    """
    # 1. Identify partitions
    group_a = [i for i, bit in enumerate(bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(bitstring) if bit == '1']
    
    # 2. Create a Linear Separation Layout
    # Group A on the left (x = -0.6), Group B on the right (x = 0.6)
    pos = {}
    for i, node in enumerate(group_a):
        pos[node] = np.array([-0.6, i - (len(group_a)-1)/2.0])
    for i, node in enumerate(group_b):
        pos[node] = np.array([0.6, i - (len(group_b)-1)/2.0])

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 3. Create a background showing Area A and Area B
    # Left side (A) is light blue, Right side (B) is a slightly different shade
    ax.axvspan(-1.5, 0, color='#a0d8f1', alpha=0.3, label='Area A (0)')
    ax.axvspan(0, 1.5, color='#e0f2fe', alpha=0.3, label='Area B (1)')

    # 4. Draw the DECISION BOUNDARY (A single, thick, red dashed line)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=4, label='Decision Boundary')

    # 5. Draw edges
    # Identify cut and uncut edges
    cut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] != bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] == bitstring[v]]

    # Draw cut edges (crossing the boundary)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, width=2.5, edge_color='black', alpha=0.9)
    # Draw uncut edges (internal to regions)
    nx.draw_networkx_edges(graph, pos, edgelist=uncut_edges, width=1.5, edge_color='gray', style='dotted', alpha=0.5)

    # 6. Draw nodes (White circles with bold labels)
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1600, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(graph, pos, font_size=22, font_family='sans-serif', font_weight='bold')

    # Add big labels for the areas
    plt.text(-0.75, 1.2, 'AREA A', fontsize=24, color='#0369a1', fontweight='bold', ha='center')
    plt.text(0.75, 1.2, 'AREA B', fontsize=24, color='#0369a1', fontweight='bold', ha='center')

    plt.title(f"Max-Cut Decision Boundary (Solution: {bitstring})", fontsize=18, pad=20)
    plt.xlim(-1.5, 1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
