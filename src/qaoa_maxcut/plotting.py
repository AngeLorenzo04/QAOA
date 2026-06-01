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
    Visualize the Max-Cut result matching the reference infographic style.
    Uses a smooth boundary that wraps around nodes to show the partition.
    """
    # 1. Fixed Square Layout as in the reference image
    pos = {0: np.array([0, 1]), 1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([1, 1])}

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 2. Create a grid for the decision boundary
    x_min, x_max = -0.6, 1.6
    y_min, y_max = -0.6, 1.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    
    # 3. Potential field to define the boundary
    # We want a boundary that separates nodes based on the bitstring
    potential = np.zeros_like(xx)
    sigma = 0.5 
    
    for i, bit in enumerate(bitstring):
        xi, yi = pos[i]
        dist_sq = (xx - xi)**2 + (yy - yi)**2
        # Sign determines the area
        weight = 1.0 if bit == '0' else -1.0
        potential += weight * np.exp(-dist_sq / (2 * sigma**2))

    # 4. Draw the smooth colored regions (Area A and Area B)
    # Using the exact colors from the reference (shades of light blue)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e1f5fe', '#81d4fa']) 
    plt.contourf(xx, yy, potential > 0, levels=[-0.5, 0.5, 1.5], cmap=cmap)

    # 5. Draw the DECISION BOUNDARY (Dashed black line as in the reference)
    plt.contour(xx, yy, potential, levels=[0], colors='black', linestyles='dashed', linewidths=2.5)

    # 6. Draw edges (Solid black lines)
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='black')

    # 7. Draw nodes (White circles with labels)
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1800, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, font_size=24, font_family='sans-serif')

    # Add Region Labels
    # Determine where to place 'A' and 'B' based on the potential field
    plt.text(-0.4, 0.8, 'A', fontsize=35, color='#01579b', fontweight='bold', alpha=0.7)
    plt.text(-0.4, 0.2, 'B', fontsize=35, color='#01579b', fontweight='bold', alpha=0.7)

    plt.title(f"Max-Cut Decision Boundary (Solution: {bitstring})", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
