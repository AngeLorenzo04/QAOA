import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

def plot_qaoa_dashboard(graph: nx.Graph, k: int, probs: np.ndarray, best_bitstring: str, 
                        node_colors: dict = None, cost_history: list = None, title: str = None) -> None:
    """
    Generate a unified 1x3 dashboard for QAOA analysis (Max-Cut or Max-k-Cut).
    
    Args:
        graph (nx.Graph): The problem graph.
        k (int): Number of partitions.
        probs (np.ndarray): Probability distribution of states.
        best_bitstring (str): The bitstring with the highest probability.
        node_colors (dict, optional): Mapping node -> color index. If None, derived from best_bitstring (for k=2).
        cost_history (list, optional): History of cost values during optimization.
        title (str, optional): Custom title for the dashboard.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    n_nodes = len(graph.nodes)
    n_qubits = len(best_bitstring)
    
    if title is None:
        title = f"QAOA Max-{k}-Cut Analysis Dashboard (Solution: {best_bitstring})"
        
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)
    
    pos = nx.spring_layout(graph, seed=42)
    
    # --- 1. Problem Graph ---
    ax1 = axes[0]
    ax1.set_title("1. Original Problem Graph", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')
    
    # --- 2. Probabilities ---
    ax2 = axes[1]
    ax2.set_title("2. Measurement Probabilities", fontsize=16)
    
    display_probs = probs
    display_labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]

    
    colors = ['tab:blue' if label == best_bitstring else 'lightgray' for label in display_labels]
    ax2.bar(display_labels, display_probs, color=colors)
    ax2.set_ylabel("Probability")
    ax2.tick_params(axis='x', rotation=45)
    
    # --- 3. Result Partition ---
    ax3 = axes[2]
    ax3.set_title(f"3. Max-{k}-Cut Partition", fontsize=16)
    
    # Determine node colors and partition groups
    if k == 2 and node_colors is None:
        # Standard Max-Cut: bit i corresponds to node i
        group_colors = ['lightblue', 'lightcoral']
        draw_colors = [group_colors[int(bit)] for bit in best_bitstring]
        partition = {i: int(bit) for i, bit in enumerate(best_bitstring)}
    else:
        # Max-k-Cut or custom colors
        cmap = plt.get_cmap('tab10')
        draw_colors = [cmap(node_colors[i]) if node_colors[i] != -1 else 'gray' for i in range(n_nodes)]
        partition = node_colors

    # Identify cut and uncut edges
    cut_edges = []
    uncut_edges = []
    for u, v in graph.edges():
        if partition[u] != partition[v] and partition[u] != -1 and partition[v] != -1:
            cut_edges.append((u, v))
        else:
            uncut_edges.append((u, v))
            
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.0, edge_color='gray', alpha=0.3)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=2.0, edge_color='darkblue', style='dashed')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, node_color=draw_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')
    
    # Legend
    legend_elements = []
    if k == 2 and node_colors is None:
        legend_elements = [
            Line2D([0], [0], color='w', marker='o', markerfacecolor='lightblue', markersize=10, label='Partition 0'),
            Line2D([0], [0], color='w', marker='o', markerfacecolor='lightcoral', markersize=10, label='Partition 1'),
        ]
    else:
        cmap = plt.get_cmap('tab10')
        legend_elements = [
            Line2D([0], [0], color='w', marker='o', markerfacecolor=cmap(i), markersize=10, label=f'Partition {i}')
            for i in range(k)
        ]
    
    legend_elements.append(Line2D([0], [0], color='darkblue', lw=2, ls='--', label='Cut Edge'))
    ax3.legend(handles=legend_elements, loc='best', fontsize=10)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
