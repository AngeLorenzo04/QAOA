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
    Create a random Erdos-Renyi graph, ensuring each node is connected to at least one other node.
    
    Args:
        n_nodes (int): Number of nodes. Must be >= 2.
        probability (float): Probability of edge creation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        nx.Graph: A NetworkX random graph with no isolated nodes.
    """
    if n_nodes < 2:
        raise ValueError("Number of nodes must be at least 2 to ensure each node is connected to at least one other node.")
        
    graph = nx.gnp_random_graph(n_nodes, probability, seed=seed)
    
    # Find isolated nodes (degree == 0)
    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    
    if isolated_nodes:
        import random
        rng = random.Random(seed)
        all_nodes = list(graph.nodes)
        
        for node in isolated_nodes:
            # Check dynamically since a previous step might have connected this node
            if graph.degree(node) == 0:
                possible_targets = [n for n in all_nodes if n != node]
                if possible_targets:
                    target = rng.choice(possible_targets)
                    graph.add_edge(node, target)
                    
    return graph

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

def is_valid_graph(graph: nx.Graph) -> bool:
    """
    Validate if a graph is considered valid.
    A graph is valid if it has at least one node and every node is connected
    to at least one other node (degree >= 1 for all nodes).
    
    Args:
        graph (nx.Graph): The graph to validate.
        
    Returns:
        bool: True if the graph is valid, False otherwise.
    """
    if len(graph.nodes) == 0:
        return False
    return all(graph.degree(node) >= 1 for node in graph.nodes)

