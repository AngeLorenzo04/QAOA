import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


def plot_graph(graph: nx.Graph) -> None:
    """
    Visualize the graph used for the Max-Cut problem.
    
    Args:
        graph: The NetworkX graph to plot.
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
    
    Args:
        probs: Array of probabilities for each basis state.
        n_wires: Number of qubits (nodes in the graph).
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
    Visualize the Max-Cut result with shaded areas for partitions and highlighted cuts.
    
    Args:
        graph: The problem graph.
        bitstring: The best solution found (e.g., '0101').
    """
    # 1. Identify partitions
    group_a = [i for i, bit in enumerate(bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(bitstring) if bit == '1']
    
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 2. Function to draw shaded areas around groups
    def draw_hull(nodes, color, label):
        if len(nodes) < 2:
            return
        
        points = np.array([pos[n] for n in nodes])
        
        # Add a small buffer/offset to the points so the hull doesn't pass exactly through node centers
        if len(nodes) == 2:
            # For 2 nodes, we create a small rectangle instead of a hull
            diff = points[1] - points[0]
            perp = np.array([-diff[1], diff[0]]) * 0.2
            hull_points = np.array([
                points[0] - perp - diff*0.1,
                points[1] - perp + diff*0.1,
                points[1] + perp + diff*0.1,
                points[0] + perp - diff*0.1
            ])
        else:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Inflate the hull slightly
            center = np.mean(points, axis=0)
            hull_points = center + (hull_points - center) * 1.4

        poly = Polygon(hull_points, alpha=0.2, color=color, label=label)
        ax.add_patch(poly)

    # Draw shaded areas
    draw_hull(group_a, 'blue', 'Group A (Partition 0)')
    draw_hull(group_b, 'red', 'Group B (Partition 1)')

    # 3. Identify and draw edges
    cut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] != bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if bitstring[u] == bitstring[v]]

    # Draw cut edges (The "Boundary")
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, width=4, 
                           edge_color='black', style='-', label='Cut Edges (Boundary)')
    # Draw uncut edges
    nx.draw_networkx_edges(graph, pos, edgelist=uncut_edges, width=1.5, 
                           edge_color='gray', alpha=0.5, label='Internal Edges')

    # 4. Draw nodes
    node_colors = ['blue' if i in group_a else 'red' for i in range(len(bitstring))]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(graph, pos, font_size=14, font_color='white', font_weight='bold')

    plt.title(f"Max-Cut Result: Bitstring {bitstring}\nClear Partitioning & Cut Boundary", fontsize=15)
    plt.legend(loc='upper right', frameon=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
