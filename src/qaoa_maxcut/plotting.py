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
    Visualize the Max-Cut result with a smooth decision boundary.
    Uses a potential field approach (Radial Basis Functions) to create a 
    smooth separation between Group A and Group B nodes.
    
    Args:
        graph: The problem graph.
        bitstring: The best solution found (e.g., '0101').
    """
    # 1. Define positions
    if len(graph.nodes) == 4:
        # Standard square layout as in the reference image
        pos = {0: np.array([0, 1]), 1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([1, 1])}
    else:
        pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 2. Create a grid for the decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # 3. Calculate a "Potential Field"
    # Nodes in Group A (0) have positive potential, Group B (1) have negative
    potential = np.zeros_like(xx)
    sigma = 0.6  # Smoothness parameter
    
    for i, bit in enumerate(bitstring):
        xi, yi = pos[i]
        dist_sq = (xx - xi)**2 + (yy - yi)**2
        weight = 1.0 if bit == '0' else -1.0
        potential += weight * np.exp(-dist_sq / (2 * sigma**2))

    # 4. Draw the smooth colored regions
    # Group A: Blue-ish, Group B: Light-blue-ish
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e0f2fe', '#a0d8f1']) # Order reversed for potential sign
    plt.contourf(xx, yy, potential > 0, levels=[-0.5, 0.5, 1.5], cmap=cmap)

    # 5. Draw the Decision Boundary (Zero-potential contour)
    plt.contour(xx, yy, potential, levels=[0], colors='black', linestyles='dashed', linewidths=2.5)

    # 6. Draw edges
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='black', alpha=0.8)

    # 7. Draw nodes (White circles as in the image)
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1500, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')

    # Add Region Labels
    plt.text(-0.35, 1.3, 'Area A', fontsize=20, color='#0369a1', fontweight='bold')
    plt.text(-0.35, -0.3, 'Area B', fontsize=20, color='#0369a1', fontweight='bold')

    plt.title(f"Max-Cut Decision Boundary (Solution: {bitstring})", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
