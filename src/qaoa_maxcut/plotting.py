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
    Visualize the Max-Cut result with colored background regions and a dashed boundary.
    This style mimics the infographic provided by the user.
    
    Args:
        graph: The problem graph.
        bitstring: The best solution found (e.g., '0101').
    """
    # 1. Define positions (Square layout for 4 nodes, or circular for more)
    if len(graph.nodes) == 4:
        pos = {0: np.array([0, 1]), 1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([1, 1])}
    else:
        pos = nx.circular_layout(graph)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 2. Create a grid to color the background based on proximity to nodes
    x_range = np.linspace(-0.5, 1.5, 200)
    y_range = np.linspace(-0.5, 1.5, 200)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Calculate which group is nearest for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    node_points = np.array([pos[i] for i in range(len(graph.nodes))])
    node_groups = np.array([int(bit) for bit in bitstring])
    
    from scipy.spatial import KDTree
    tree = KDTree(node_points)
    _, nearest_node_idx = tree.query(grid_points)
    grid_groups = node_groups[nearest_node_idx].reshape(xx.shape)

    # 3. Draw the colored background regions
    from matplotlib.colors import ListedColormap
    # Light blue for Group A (0), Lighter blue for Group B (1)
    cmap = ListedColormap(['#a0d8f1', '#e0f2fe']) 
    plt.contourf(xx, yy, grid_groups, levels=[-0.5, 0.5, 1.5], cmap=cmap)

    # 4. Draw the dashed boundary line (The "Cut")
    plt.contour(xx, yy, grid_groups, levels=[0.5], colors='black', linestyles='dashed', linewidths=2)

    # 5. Draw edges
    # For this visualization, all edges are drawn similarly, but the boundary line shows the cut
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='black', alpha=0.7)

    # 6. Draw nodes (white circles with labels as in the image)
    nx.draw_networkx_nodes(graph, pos, node_color='white', node_size=1200, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, font_size=18, font_family='sans-serif', font_weight='normal')

    # Add labels A and B to the regions
    plt.text(-0.3, 0.7, 'A', fontsize=30, alpha=0.6, fontweight='bold')
    plt.text(-0.3, 0.2, 'B', fontsize=30, alpha=0.6, fontweight='bold')

    plt.title(f"Max-Cut Solution: {bitstring}", fontsize=15, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
