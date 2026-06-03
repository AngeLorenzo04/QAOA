import pennylane as qml
import networkx as nx


def apply_qaoa_layer(gamma: float, beta: float, cost_h, mixer_h) -> None:
    """
    Applica un singolo layer QAOA composto da un operatore di costo e un operatore mixer.
    
    Matematicamente, questo applica l'evoluzione unitaria e^{-i * beta * H_M} e^{-i * gamma * H_C}
    allo stato quantistico.

    Args:
        gamma (float): Il parametro variazionale per l'Hamiltoniana di costo.
        beta (float): Il parametro variazionale per l'Hamiltoniana mixer.
        cost_h: L'Hamiltoniana di costo che codifica il problema Max-Cut.
        mixer_h: L'Hamiltoniana mixer utilizzata per esplorare lo spazio degli stati.
    """
    # Applica l'evoluzione temporale dell'Hamiltoniana di costo
    qml.qaoa.cost_layer(gamma, cost_h)
    # Applica l'evoluzione temporale dell'Hamiltoniana mixer
    qml.qaoa.mixer_layer(beta, mixer_h)


def create_qaoa_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Crea un circuito quantistico (QNode) che valuta il valore di aspettativa dell'Hamiltoniana di costo.
    
    Questo circuito viene utilizzato durante il ciclo di ottimizzazione classica. L'ottimizzatore
    regola i gamma e i beta per minimizzare il valore di aspettativa (che corrisponde a massimizzare il taglio).

    Args:
        graph (nx.Graph): Il grafo che rappresenta l'istanza Max-Cut.
        cost_h: L'Hamiltoniana di costo.
        mixer_h: L'Hamiltoniana mixer.
    
    Returns:
        function: Un QNode di PennyLane che accetta parametri variazionali e restituisce un costo scalare.
    """
    n_wires = len(graph.nodes)
    # Inizializza un simulatore quantistico locale con un wire per ogni nodo nel grafo
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        """
        Il circuito quantistico eseguibile.
        Args:
            params: Un array 2D dove params[0] sono i gamma e params[1] sono i beta.
        """
        # Step 1: Inizializzazione. Crea una sovrapposizione equa di tutti i possibili stati (tagli).
        # Questo viene fatto applicando una porta Hadamard a ogni qubit.
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Step 2: Applica p layer di QAOA.
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # Step 3: Misurazione. Restituisce il valore di aspettativa dell'Hamiltoniana di costo.
        return qml.expval(cost_h)

    return circuit


def create_sampling_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Crea un circuito quantistico (QNode) che restituisce la distribuzione di probabilità di tutti gli stati di base.
    
    Questo circuito viene utilizzato *dopo* l'ottimizzazione per identificare le stringhe di bit più probabili,
    che rappresentano le nostre migliori stime per la partizione Max-Cut ottimale.

    Args:
        graph (nx.Graph): Il grafo che rappresenta l'istanza Max-Cut.
        cost_h: L'Hamiltoniana di costo.
        mixer_h: L'Hamiltoniana mixer.
    
    Returns:
        function: Un QNode di PennyLane che accetta parametri ottimizzati e restituisce un array di probabilità.
    """
    n_wires = len(graph.nodes)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        # Prepara lo stato di sovrapposizione equa
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Applica i layer QAOA ottimizzati
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # Invece del valore di aspettativa, restituisce la probabilità di misurare ogni possibile stringa di bit
        return qml.probs(wires=range(n_wires))

    return circuit
