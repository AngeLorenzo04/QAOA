import networkx as nx


def create_cycle_graph() -> nx.Graph:
    """Create a simple 4-node cycle graph for the Max-Cut demo."""
    return nx.Graph([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])