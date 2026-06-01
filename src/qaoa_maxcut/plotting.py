import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_dashboard(graph: nx.Graph, probs: np.ndarray, best_bitstring: str, cost_history: list, n_wires: int) -> None:
    """
    Generate a unified 1x3 dashboard for the Max-Cut QAOA demo.
    
    This function creates a single Matplotlib window containing three subplots:
    1. The original problem graph geometry.
    2. The quantum measurement probabilities after QAOA optimization.
    3. The final Max-Cut partition highlighting the cut edges.
    
    Args:
        graph (nx.Graph): The original NetworkX problem graph.
        probs (np.ndarray): Array of measurement probabilities for each of the 2^n basis states.
        best_bitstring (str): The bitstring (e.g., '0101') with the highest probability.
        cost_history (list): A list tracking the expectation value of the cost Hamiltonian at each step.
        n_wires (int): The number of qubits, which corresponds to the number of nodes in the graph.
    """
    # Apply a modern, clean style suitable for academic presentations and thesis figures.
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a 1x3 figure layout (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"QAOA Max-Cut Analysis Dashboard (Solution: {best_bitstring})", fontsize=20, fontweight='bold', y=1.05)

    # Calculate a standard spring layout for the graph nodes to use consistently across subplots 1 and 3.
    # A fixed seed ensures the layout looks the same every time the code runs.
    pos = nx.spring_layout(graph, seed=42)

    # ==========================================
    # --- Subplot 1: Original Problem Graph ---
    # ==========================================
    ax1 = axes[0]
    ax1.set_title("1. Original Problem Graph", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')

    # ===============================================
    # --- Subplot 2: Measurement Probabilities ---
    # ===============================================
    ax2 = axes[1]
    ax2.set_title("2. Measurement Probabilities", fontsize=16)
    
    # Generate labels for the X-axis (all possible bitstrings like '00', '01', '10', '11')
    bitstrings = [format(i, f'0{n_wires}b') for i in range(2**n_wires)]
    
    # Highlight the bar corresponding to the best solution found by the algorithm
    colors = ['tab:blue' if bs == best_bitstring else 'lightgray' for bs in bitstrings]
    
    ax2.bar(bitstrings, probs, color=colors)
    ax2.set_xlabel("Bitstrings (Basis States)", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # =======================================
    # --- Subplot 3: Max-Cut Partition ---
    # =======================================
    ax3 = axes[2]
    ax3.set_title("3. Max-Cut Partition", fontsize=16)

    # Step 3.1: Separate nodes into two groups based on the bitstring solution (0 vs 1).
    group_a = [i for i, bit in enumerate(best_bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(best_bitstring) if bit == '1']
    
    # Step 3.2: Separate edges into "cut" and "uncut".
    # An edge is "cut" if its two endpoints belong to different groups (different bits).
    cut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] != best_bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] == best_bitstring[v]]

    # Step 3.3: Draw the uncut edges (faded, solid lines)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.5, edge_color='gray', style='solid', alpha=0.5)
    
    # Step 3.4: Draw the cut edges (prominent, dark blue, dashed lines representing the "Boundary")
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=1.5, edge_color='darkblue', style='dashed')

    # Step 3.5: Draw the nodes, coloring them differently for each partition group
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_a, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_b, node_color='lightcoral', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')

    # Step 3.6: Add a custom legend to explain the colors and edge styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightblue', markersize=12, label='Partition 0', markeredgecolor='black'),
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightcoral', markersize=12, label='Partition 1', markeredgecolor='black'),
        Line2D([0], [0], color='darkblue', lw=1.5, ls='--', label='Cut Edge')
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=10)
    ax3.axis('off')

    # Ensure everything fits nicely before displaying
    plt.tight_layout()
    plt.show()
