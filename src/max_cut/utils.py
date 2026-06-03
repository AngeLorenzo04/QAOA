import networkx as nx


def create_cycle_graph() -> nx.Graph:
    """
    Crea un semplice grafo a ciclo di 4 nodi (un quadrato) per la demo di Max-Cut.
    
    Nel problema Max-Cut, l'obiettivo è partizionare questi nodi in due 
    set distinti tali che il numero di archi che collegano i due set sia massimizzato.
    Per un ciclo di 4 nodi, il taglio ottimale comporta il taglio di tutti i 4 archi.

    Returns:
        nx.Graph: Un oggetto grafo NetworkX che rappresenta la geometria del problema.
    """
    # Crea un grafo e aggiunge archi definendo un ciclo chiuso: 0-1-2-3-0
    return nx.Graph([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])
