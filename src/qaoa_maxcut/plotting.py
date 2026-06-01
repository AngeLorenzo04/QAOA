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
    Visualize the Max-Cut result by calculating a mathematical decision boundary.
    Displays the explicit RBF equation f(x,y) = 0 used to separate the classes.
    """
    X = np.array([
        [0, 1], # Node 0
        [0, 0], # Node 1
        [1, 0], # Node 2
        [1, 1]  # Node 3
    ])
    
    weights = np.array([1.0 if bit == '0' else -1.0 for bit in bitstring])

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    
    f_xy = np.zeros_like(xx)
    sigma = 0.5 
    sigma_sq2 = 2 * sigma**2
    
    # Build the LaTeX string for the function
    equation_parts = []
    for i in range(len(X)):
        xi, yi = X[i]
        dist_sq = (xx - xi)**2 + (yy - yi)**2
        w = weights[i]
        f_xy += w * np.exp(-dist_sq / sigma_sq2)
        
        # Format the term for the equation
        sign = "+" if w > 0 else "-"
        term = rf"{sign} e^{{-\frac{{(x - {xi})^2 + (y - {yi})^2}}{{{sigma_sq2}}}_{{}}}}"
        equation_parts.append(term)

    # Combine parts into a single LaTeX equation
    equation_str = "$f(x, y) = " + " ".join(equation_parts) + " = 0$"

    plt.figure(figsize=(12, 10)) # Slightly taller to fit the equation
    ax = plt.gca()

    plt.contourf(xx, yy, f_xy > 0, levels=[-0.5, 0.5, 1.5], colors=['#e0f2fe', '#a0d8f1'], alpha=0.6)

    # Draw the Decision Boundary
    plt.contour(xx, yy, f_xy, levels=[0], colors='red', linewidths=4, linestyles='solid')

    pos = {i: X[i] for i in range(len(X))}
    
    # Highlight edges
    for u, v in graph.edges():
        is_cut = bitstring[u] != bitstring[v]
        color = '#d32f2f' if is_cut else 'black'
        width = 4 if is_cut else 2
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color, lw=width)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1800, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(graph, pos, font_size=24, font_family='sans-serif', font_weight='bold')

    plt.text(-0.3, 1.3, 'Area A (+)', fontsize=20, color='#01579b', fontweight='bold')
    plt.text(-0.3, -0.3, 'Area B (-)', fontsize=20, color='#01579b', fontweight='bold')
    
    # Add the MATHEMATICAL EQUATION to the plot
    plt.text(0.5, -0.65, equation_str, fontsize=14, ha='center', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.title(f"Max-Cut Decision Boundary\nMathematical separator for solution {bitstring}", fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # Adjust layout to fit everything
    plt.subplots_adjust(bottom=0.2)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Decision Boundary: f(x,y)=0'),
        Line2D([0], [0], color='#d32f2f', lw=4, label='Cut Edge')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()
