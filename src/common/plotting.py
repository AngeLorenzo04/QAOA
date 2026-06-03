import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_maxcut_dashboard(graph: nx.Graph, probs: np.ndarray, best_bitstring: str, cost_history: list) -> None:
    """
    Generate a unified 1x3 dashboard for the Max-Cut QAOA demo.
    """
    n_wires = len(graph.nodes)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"QAOA Max-Cut Analysis Dashboard (Solution: {best_bitstring})", fontsize=20, fontweight='bold', y=1.05)

    pos = nx.spring_layout(graph, seed=42)

    # Subplot 1: Original Graph
    ax1 = axes[0]
    ax1.set_title("1. Original Problem Graph", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')

    # Subplot 2: Probabilities
    ax2 = axes[1]
    ax2.set_title("2. Measurement Probabilities", fontsize=16)
    bitstrings = [format(i, f'0{n_wires}b') for i in range(2**n_wires)]
    colors = ['tab:blue' if bs == best_bitstring else 'lightgray' for bs in bitstrings]
    
    ax2.bar(bitstrings, probs, color=colors)
    ax2.set_xlabel("Bitstrings", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # Subplot 3: Partition
    ax3 = axes[2]
    ax3.set_title("3. Max-Cut Partition", fontsize=16)
    
    group_a = [i for i, bit in enumerate(best_bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(best_bitstring) if bit == '1']
    
    cut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] != best_bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] == best_bitstring[v]]

    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.5, edge_color='gray', style='solid', alpha=0.5)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=1.5, edge_color='darkblue', style='dashed')

    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_a, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_b, node_color='lightcoral', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')

    plt.tight_layout()
    plt.show()
