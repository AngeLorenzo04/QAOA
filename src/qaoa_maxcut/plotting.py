import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(graph: nx.Graph) -> None:
    """
    Visualize the graph used for the Max-Cut problem.
    """
    pos = {0: np.array([0, 1]), 1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([1, 1])}
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
    Visualize the Max-Cut result with a continuous decision boundary 
    that physically cuts the edges.
    """
    # 1. Fixed Square Layout
    pos = {0: np.array([0, 1]), 1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([1, 1])}

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 2. Identify partitions
    group_a = [i for i, bit in enumerate(bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(bitstring) if bit == '1']
    
    # 3. Create a smooth field to define the boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    
    potential = np.zeros_like(xx)
    sigma = 0.5
    for i, bit in enumerate(bitstring):
        xi, yi = pos[i]
        dist_sq = (xx - xi)**2 + (yy - yi)**2
        weight = 1.0 if bit == '0' else -1.0
        potential += weight * np.exp(-dist_sq / (2 * sigma**2))

    # 4. Color the two areas (distinctly but lightly)
    ax.contourf(xx, yy, potential > 0, levels=[-0.5, 0.5, 1.5], colors=['#e0f2fe', '#a0d8f1'], alpha=0.5)

    # 5. Draw the CONTINUOUS DECISION BOUNDARY
    # This line explicitly separates the nodes and intersects the edges
    ax.contour(xx, yy, potential, levels=[0], colors='red', linewidths=3, linestyles='solid')

    # 6. Draw edges with highlight on the ones being cut
    for u, v in graph.edges():
        color = 'black'
        width = 2
        style = 'solid'
        
        # If the edge is cut, we can make it more prominent to show the intersection
        if bitstring[u] != bitstring[v]:
            width = 4
            color = '#d32f2f' # Dark red for cut edges
        
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-", color=color, lw=width, ls=style))

    # 7. Draw nodes as white circles
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(graph, pos, font_size=24, font_family='sans-serif', font_weight='bold')

    # Add labels
    plt.text(-0.3, 1.3, 'Group A', fontsize=20, color='#0369a1', fontweight='bold')
    plt.text(-0.3, -0.3, 'Group B', fontsize=20, color='#0369a1', fontweight='bold')
    
    # Legend for the "Cut"
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Decision Boundary (The Cut)'),
        Line2D([0], [0], color='#d32f2f', lw=4, label='Cut Edges')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title(f"MAX-CUT: The Decision Boundary intersects the edges\nSolution: {bitstring}", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
