import networkx as nx

def create_cycle_graph(n_nodes: int = 3) -> nx.Graph:
    """Crea un grafo a ciclo."""
    return nx.cycle_graph(n_nodes)

def create_complete_graph(n_nodes: int = 3) -> nx.Graph:
    """Crea un grafo completo."""
    return nx.complete_graph(n_nodes)

def create_random_graph(n_nodes: int = 4, probability: float = 0.5) -> nx.Graph:
    """Crea un grafo casuale di Erdos-Renyi."""
    return nx.gnp_random_graph(n_nodes, probability, seed=42)

def create_star_graph(n_nodes: int = 4) -> nx.Graph:
    """Crea un grafo a stella."""
    return nx.star_graph(n_nodes - 1)
