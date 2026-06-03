import networkx as nx

def create_cycle_graph(n_nodes: int = 4) -> nx.Graph:
    """
    Create a simple cycle graph.
    
    Args:
        n_nodes (int): Number of nodes in the cycle.
        
    Returns:
        nx.Graph: A NetworkX cycle graph.
    """
    return nx.cycle_graph(n_nodes)

def create_complete_graph(n_nodes: int = 4) -> nx.Graph:
    """
    Create a complete graph (all-to-all connectivity).
    
    Args:
        n_nodes (int): Number of nodes.
        
    Returns:
        nx.Graph: A NetworkX complete graph.
    """
    return nx.complete_graph(n_nodes)

def create_random_graph(n_nodes: int = 5, probability: float = 0.5, seed: int = 42) -> nx.Graph:
    """
    Create a random Erdos-Renyi graph.
    
    Args:
        n_nodes (int): Number of nodes.
        probability (float): Probability of edge creation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        nx.Graph: A NetworkX random graph.
    """
    return nx.gnp_random_graph(n_nodes, probability, seed=seed)

def create_petersen_graph() -> nx.Graph:
    """
    Create the Petersen graph (10 nodes, 15 edges).
    
    Returns:
        nx.Graph: A NetworkX Petersen graph.
    """
    return nx.petersen_graph()

def create_star_graph(n_nodes: int = 4) -> nx.Graph:
    """
    Create a star graph.
    
    Args:
        n_nodes (int): Total number of nodes (1 center + n-1 leaves).
        
    Returns:
        nx.Graph: A NetworkX star graph.
    """
    return nx.star_graph(n_nodes - 1)

def create_triangle_graph() -> nx.Graph:
    """
    Create a 3-node complete graph (triangle).
    
    Returns:
        nx.Graph: A NetworkX graph object.
    """
    return nx.complete_graph(3)
