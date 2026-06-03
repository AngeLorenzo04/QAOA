import networkx as nx

def create_cycle_graph(n_nodes: int = 4) -> nx.Graph:
    """
    Create a simple cycle graph (a regular polygon).
    
    Args:
        n_nodes (int): Number of nodes in the cycle.
        
    Returns:
        nx.Graph: A NetworkX graph object.
    """
    return nx.cycle_graph(n_nodes)

def create_triangle_graph() -> nx.Graph:
    """
    Create a 3-node complete graph (triangle).
    
    Returns:
        nx.Graph: A NetworkX graph object.
    """
    return nx.complete_graph(3)
