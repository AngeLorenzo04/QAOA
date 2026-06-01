import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_dashboard(graph: nx.Graph, probs: np.ndarray, best_bitstring: str, cost_history: list, n_wires: int) -> None:
    """
    Generate a unified 1x3 dashboard for the Max-Cut QAOA demo.
    
    Args:
        graph: The original problem graph.
        probs: Array of probabilities for each bitstring state.
        best_bitstring: The most probable solution found.
        cost_history: List of expectation values during optimization.
        n_wires: Number of qubits/nodes.
    """
    # Apply modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"QAOA Max-Cut Analysis Dashboard (Solution: {best_bitstring})", fontsize=20, fontweight='bold', y=1.05)

    # Common layout for the graphs
    # We use a standard layout, not restricted to Cartesian coordinate visuals
    pos = nx.spring_layout(graph, seed=42)

    # --- Subplot 1: Problem Graph ---
    ax1 = axes[0]
    ax1.set_title("1. Original Problem Graph", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')

    # --- Subplot 2: Measurement Probabilities ---
    ax2 = axes[1]
    ax2.set_title("2. Measurement Probabilities", fontsize=16)
    bitstrings = [format(i, f'0{n_wires}b') for i in range(2**n_wires)]
    
    # Highlight the best bitstring
    colors = ['tab:blue' if bs == best_bitstring else 'lightgray' for bs in bitstrings]
    ax2.bar(bitstrings, probs, color=colors)
    ax2.set_xlabel("Bitstrings (Basis States)", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # --- Subplot 3: Max-Cut Result ---
    ax3 = axes[2]
    ax3.set_title("3. Max-Cut Partition", fontsize=16)

    # 1. Identify partitions
    group_a = [i for i, bit in enumerate(best_bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(best_bitstring) if bit == '1']
    
    # 2. Identify edges
    cut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] != best_bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] == best_bitstring[v]]

    # 3. Draw uncut edges (light, solid)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.5, edge_color='gray', style='solid', alpha=0.5)
    
    # 4. Draw cut edges (thin, dark blue, dashed - per user request)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=1.5, edge_color='darkblue', style='dashed')

    # 5. Draw nodes (colored by partition)
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_a, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_b, node_color='lightcoral', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')

    # Legend for Plot 3
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightblue', markersize=12, label='Partition 0', markeredgecolor='black'),
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightcoral', markersize=12, label='Partition 1', markeredgecolor='black'),
        Line2D([0], [0], color='darkblue', lw=1.5, ls='--', label='Cut Edge')
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=10)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
