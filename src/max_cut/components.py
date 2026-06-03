import pennylane as qml
import networkx as nx


def build_maxcut_hamiltonians(graph: nx.Graph):
    """
    Costruisce le Hamiltoniane di Costo e Mixer per il problema Max-Cut.

    - Hamiltoniana di Costo (H_C): Codifica la funzione obiettivo del problema. 
      Il suo stato fondamentale corrisponde alla soluzione Max-Cut ottimale.
    - Hamiltoniana Mixer (H_M): Responsabile dell'esplorazione dello spazio delle soluzioni 
      guidando le transizioni quantistiche tra diversi stati.

    Args:
        graph (nx.Graph): Il grafo del problema.

    Returns:
        tuple: Una tupla contenente (cost_hamiltonian, mixer_hamiltonian).
    """
    # PennyLane fornisce una funzione integrata per generare queste Hamiltoniane 
    # direttamente da un grafo NetworkX per il problema Max-Cut.
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    return cost_h, mixer_h
