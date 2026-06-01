import pennylane as qml
import networkx as nx


def apply_qaoa_layer(gamma: float, beta: float, cost_h, mixer_h) -> None:
    """
    Apply one QAOA layer consisting of the cost and mixer operators.
    
    Args:
        gamma: Parameter for the cost Hamiltonian evolution.
        beta: Parameter for the mixer Hamiltonian evolution.
        cost_h: The cost Hamiltonian (problem-specific).
        mixer_h: The mixer Hamiltonian (e.g., sum of PauliX).
    """
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)


def create_qaoa_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Create a QNode that evaluates the expectation value of the cost Hamiltonian.
    
    This version supports multiple layers (p >= 1) by iterating over arrays of 
    gamma and beta parameters.

    Args:
        graph: The graph representing the Max-Cut instance.
        cost_h: The cost Hamiltonian.
        mixer_h: The mixer Hamiltonian.
    
    Returns:
        A QNode that takes a list of parameters [[gamma1, gamma2, ...], [beta1, beta2, ...]]
        and returns the expectation value of the cost Hamiltonian.
    """
    n_wires = len(graph.nodes)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        # Initial state: superposition of all basis states
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Apply p layers of QAOA
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # We return the expectation value of the cost Hamiltonian to optimize
        return qml.expval(cost_h)

    return circuit


def create_sampling_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Create a QNode that returns the probability distribution of the states.
    
    This is used after optimization to identify the most probable solutions (cuts).

    Args:
        graph: The graph representing the Max-Cut instance.
        cost_h: The cost Hamiltonian.
        mixer_h: The mixer Hamiltonian.
    
    Returns:
        A QNode that takes the optimized parameters and returns probabilities for all 2^n states.
    """
    n_wires = len(graph.nodes)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        # Initial state: superposition of all basis states
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Apply p layers of QAOA
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # Return the probability of each bitstring (basis state)
        return qml.probs(wires=range(n_wires))

    return circuit
