import networkx as nx


def create_cycle_graph() -> nx.Graph:
    """
    Create a simple 4-node cycle graph (a square) for the Max-Cut demo.
    
    In the Max-Cut problem, the goal is to partition these nodes into two 
    distinct sets such that the number of edges connecting the two sets is maximized.
    For a 4-node cycle, the optimal cut involves cutting all 4 edges.

    Returns:
        nx.Graph: A NetworkX graph object representing the problem geometry.
    """
    # Create a graph and add edges defining a closed loop: 0-1-2-3-0
    return nx.Graph([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])
