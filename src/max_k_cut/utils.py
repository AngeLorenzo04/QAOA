import networkx as nx

def create_cycle_graph(n_nodes: int = 3) -> nx.Graph:
    """
    Crea un grafo a ciclo con un numero specificato di nodi.
    
    Args:
        n_nodes (int): Numero di nodi nel ciclo.
        
    Returns:
        nx.Graph: Un oggetto grafo NetworkX.
    """
    return nx.cycle_graph(n_nodes)

def create_complete_graph(n_nodes: int = 3) -> nx.Graph:
    """
    Crea un grafo completo (clique) dove ogni nodo è connesso a tutti gli altri.
    
    Args:
        n_nodes (int): Numero di nodi.
        
    Returns:
        nx.Graph: Un oggetto grafo NetworkX.
    """
    return nx.complete_graph(n_nodes)
