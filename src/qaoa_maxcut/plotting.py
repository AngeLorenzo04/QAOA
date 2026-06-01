import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(graph: nx.Graph) -> None:
    """Plot the input graph for the Max-Cut demo."""
    pos = nx.circular_layout(graph)

    plt.figure(figsize=(5, 5))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=900,
        font_weight="bold",
        edgecolors="black",
    )
    plt.title("Max-Cut demo graph")
    plt.tight_layout()
    plt.show()