import pennylane as qml
import networkx as nx
import numpy as np

def build_max_k_cut_hamiltonians(graph: nx.Graph, k: int, penalty_weight: float = 1.0):
    """
    Costruisce l'Hamiltoniana di Costo (con penalità) e l'Hamiltoniana Mixer per Max-k-Cut.
    
    Il problema Max-k-Cut viene mappato su n*k qubit, dove n è il numero di nodi.
    Ogni nodo i è rappresentato da k qubit; il qubit (i, s) è 1 se il nodo i ha colore s.

    Args:
        graph (nx.Graph): Il grafo del problema.
        k (int): Numero di colori/partizioni.
        penalty_weight (float): Peso della penalità per il vincolo "un solo colore per nodo".

    Returns:
        tuple: (cost_h, mixer_h)
    """
    n_nodes = len(graph.nodes)
    n_qubits = n_nodes * k
    
    coeffs = []
    obs = []
    
    # 1. Obiettivo: Penalizza nodi adiacenti con lo stesso colore
    # Termine: sum_{(u,v) in E} sum_s n_{u,s} * n_{v,s}
    for u, v in graph.edges():
        for s in range(k):
            w_u, w_v = u * k + s, v * k + s
            # n_u,s * n_v,s = (1-Z_u)/2 * (1-Z_v)/2 = 1/4 (1 - Z_u - Z_v + Z_u Z_v)
            coeffs.extend([0.25, -0.25, -0.25, 0.25])
            obs.extend([
                qml.Identity(w_u),
                qml.PauliZ(w_u),
                qml.PauliZ(w_v),
                qml.PauliZ(w_u) @ qml.PauliZ(w_v)
            ])

    # 2. Penalità: Penalizza i nodi che non hanno esattamente un colore assegnato
    # Termine di penalità: alpha * sum_i (sum_s n_{i,s} - 1)^2
    # Espandendo: (sum_s n_{i,s} - 1)^2 = sum_{s1!=s2} n_{i,s1} n_{i,s2} - sum_s n_{i,s} + 1
    
    alpha = penalty_weight
    for i in range(n_nodes):
        # sum_{s1!=s2} n_{i,s1} n_{i,s2}
        for s1 in range(k):
            for s2 in range(s1 + 1, k):
                w1, w2 = i * k + s1, i * k + s2
                # n_i,s1 * n_i,s2 = 1/4 (1 - Z_1 - Z_2 + Z_1 Z_2)
                coeffs.extend([alpha*0.25, -alpha*0.25, -alpha*0.25, alpha*0.25])
                obs.extend([
                    qml.Identity(w1),
                    qml.PauliZ(w1),
                    qml.PauliZ(w2),
                    qml.PauliZ(w1) @ qml.PauliZ(w2)
                ])
        
        # - sum_s n_{i,s}
        for s in range(k):
            w = i * k + s
            # -n_i,s = -(1-Z)/2 = -0.5 + 0.5Z
            coeffs.extend([-alpha*0.5, alpha*0.5])
            obs.extend([qml.Identity(w), qml.PauliZ(w)])
            
        # + 1 (costante per ogni nodo i)
        coeffs.append(alpha)
        obs.append(qml.Identity(i * k))

    cost_h = qml.Hamiltonian(coeffs, obs)

    # 3. Mixer Standard X
    # Applica rotazioni X su tutti i qubit per esplorare lo spazio degli stati.
    mixer_h = qml.qaoa.x_mixer(range(n_qubits))

    return cost_h, mixer_h
