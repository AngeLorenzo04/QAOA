import pennylane as qml
import networkx as nx

def create_max_k_cut_circuit(graph: nx.Graph, k: int, cost_h, mixer_h):
    """
    Crea un circuito QAOA per Max-k-Cut utilizzando il metodo di penalità.
    
    Args:
        graph (nx.Graph): Il grafo del problema.
        k (int): Numero di partizioni.
        cost_h: Hamiltoniana di costo (include le penalità).
        mixer_h: Hamiltoniana mixer.
        
    Returns:
        function: QNode di PennyLane per l'ottimizzazione.
    """
    n_nodes = len(graph.nodes)
    n_qubits = n_nodes * k
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        # Step 1: Inizializzazione
        # Inizia in una sovrapposizione equa di tutti gli stati
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Step 2: Layer QAOA
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)

        return qml.expval(cost_h)

    return circuit

def create_max_k_cut_sampling_circuit(graph: nx.Graph, k: int, cost_h, mixer_h):
    """
    Crea un circuito di campionamento per Max-k-Cut.
    
    Returns:
        function: QNode di PennyLane che restituisce le probabilità.
    """
    n_nodes = len(graph.nodes)
    n_qubits = n_nodes * k
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(beta, mixer_h)

        return qml.probs(wires=range(n_qubits))

    return circuit
