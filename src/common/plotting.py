import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

def plot_qaoa_dashboard(graph: nx.Graph, k: int, probs: np.ndarray, best_bitstring: str, 
                        node_colors: dict = None, cost_history: list = None, trajectory_params: list = None, title: str = None) -> None:
    """
    Generate a unified 1x3 dashboard for QAOA analysis (Max-Cut or Max-k-Cut).
    
    Args:
        graph (nx.Graph): The problem graph.
        k (int): Number of partitions.
        probs (np.ndarray): Probability distribution of states.
        best_bitstring (str): The bitstring with the highest probability.
        node_colors (dict, optional): Mapping node -> color index. If None, derived from best_bitstring (for k=2).
        cost_history (list, optional): History of cost values during optimization.
        trajectory_params (list, optional): History of parameter values during optimization.
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
    
    # --- 1. Probabilities ---
    ax1 = axes[0]
    
    all_labels = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    sorted_pairs = sorted(zip(all_labels, probs), key=lambda x: x[1], reverse=True)
    
    if len(sorted_pairs) > 16:
        display_pairs = sorted_pairs[:16]
        ax1.set_title("1. Measurement Probabilities (Top 16)", fontsize=14, fontweight='bold', pad=10)
    else:
        display_pairs = sorted_pairs
        ax1.set_title("1. Measurement Probabilities (Sorted)", fontsize=14, fontweight='bold', pad=10)
        
    display_labels = [x[0] for x in display_pairs]
    display_probs = [x[1] for x in display_pairs]
    
    colors = ['tab:blue' if label == best_bitstring else 'lightgray' for label in display_labels]
    ax1.bar(display_labels, display_probs, color=colors)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # --- 2. Result Partition ---
    ax2 = axes[1]
    
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
            
    exact_val = graph.graph.get('exact_max_cut_value', -1)
    if exact_val == -1 and k == 2 and len(graph.nodes) <= 16:
        try:
            from src.data.exact_maxcut_solver import find_exact_maxcut
            exact_val = find_exact_maxcut(graph)['max_cut_value']
        except Exception:
            pass
            
    expected_cost = None
    if cost_history:
        if isinstance(cost_history, dict) and 'fun' in cost_history and cost_history['fun']:
            expected_cost = cost_history['fun'][-1]
        elif isinstance(cost_history, (list, np.ndarray)) and len(cost_history) > 0:
            expected_cost = cost_history[-1]

    best_measured_cut = len(cut_edges)
    if k == 2:
        for state_int, prob in enumerate(probs):
            if prob > 1e-6:
                bstr = format(state_int, f'0{n_qubits}b')
                c = sum(1 for u, v in graph.edges() if bstr[u] != bstr[v])
                if c > best_measured_cut:
                    best_measured_cut = c

    if exact_val != -1 and exact_val is not None and exact_val > 0:
        ratio = len(cut_edges) / exact_val
        ax2.set_title(f"2. Max-{k}-Cut Partition\n(Taglio Mostrato: {len(cut_edges)}/{exact_val}, Ratio: {ratio:.4f}\nCosto Soluzione Migliore: {-best_measured_cut}, Costo Mostrato: {-len(cut_edges)})", fontsize=14, fontweight='bold', pad=10)
    else:
        ax2.set_title(f"2. Max-{k}-Cut Partition\n(Rami tagliati: {len(cut_edges)}\nCosto Soluzione Migliore: {-best_measured_cut}, Costo Mostrato: {-len(cut_edges)})", fontsize=14, fontweight='bold', pad=10)

    nx.draw_networkx_edges(graph, pos, ax=ax2, edgelist=uncut_edges, width=1.0, edge_color='gray', alpha=0.3)
    nx.draw_networkx_edges(graph, pos, ax=ax2, edgelist=cut_edges, width=2.0, edge_color='darkblue', style='dashed')
    nx.draw_networkx_nodes(graph, pos, ax=ax2, node_color=draw_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax2, font_size=14, font_weight='bold')
    
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
    ax2.legend(handles=legend_elements, loc='best', fontsize=10)
    ax2.axis('off')

    # --- 3. Parameter Evolution ---
    ax3 = axes[2]
    
    if trajectory_params is not None and len(trajectory_params) > 0:
        params_array = np.array(trajectory_params)
        n_params = params_array.shape[1]
        p = n_params // 2
        
        iterations = range(len(trajectory_params))
        
        # Plot Betas (H_M)
        for i in range(p):
            ax3.plot(iterations, params_array[:, i], label=f'$\\beta_{{{i+1}}}$ ($H_M$ mixer)', linestyle='--')
            
        # Plot Gammas (H_C)
        for i in range(p):
            ax3.plot(iterations, params_array[:, p+i], label=f'$\\gamma_{{{i+1}}}$ ($H_C$ cost)', linestyle='-')
            
        ax3.set_xlabel('Iterazione', fontsize=12)
        ax3.set_ylabel('Angolo (rad)', fontsize=12)
        ax3.set_title("3. Variazione parametri $\\gamma$ e $\\beta$", fontsize=14, fontweight='bold', pad=10)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='best', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Nessun dato sulla traiettoria', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.axis('off')
        ax3.set_title("3. Variazione parametri $\\gamma$ e $\\beta$", fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.show()
