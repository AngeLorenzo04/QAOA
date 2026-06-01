import pennylane as qml
import networkx as nx


def apply_qaoa_layer(gamma: float, beta: float, cost_h, mixer_h) -> None:
    """
    Apply a single QAOA layer consisting of a cost operator and a mixer operator.
    
    Mathematically, this applies the unitary evolution e^{-i * beta * H_M} e^{-i * gamma * H_C}
    to the quantum state.

    Args:
        gamma (float): The variational parameter for the cost Hamiltonian.
        beta (float): The variational parameter for the mixer Hamiltonian.
        cost_h: The cost Hamiltonian encoding the Max-Cut problem.
        mixer_h: The mixer Hamiltonian used to explore the state space.
    """
    # Apply the time evolution of the cost Hamiltonian
    qml.qaoa.cost_layer(gamma, cost_h)
    # Apply the time evolution of the mixer Hamiltonian
    qml.qaoa.mixer_layer(beta, mixer_h)


def create_qaoa_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Create a quantum circuit (QNode) that evaluates the expectation value of the cost Hamiltonian.
    
    This circuit is used during the classical optimization loop. The optimizer adjusts 
    the gammas and betas to minimize the expectation value (which corresponds to maximizing the cut).

    Args:
        graph (nx.Graph): The graph representing the Max-Cut instance.
        cost_h: The cost Hamiltonian.
        mixer_h: The mixer Hamiltonian.
    
    Returns:
        function: A PennyLane QNode that takes variational parameters and returns a scalar cost.
    """
    n_wires = len(graph.nodes)
    # Initialize a local quantum simulator with a wire for each node in the graph
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        """
        The executable quantum circuit.
        Args:
            params: A 2D array where params[0] are gammas and params[1] are betas.
        """
        # Step 1: Initialization. Create an equal superposition of all possible states (cuts).
        # This is done by applying a Hadamard gate to every qubit.
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Step 2: Apply p layers of QAOA.
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # Step 3: Measurement. Return the expectation value of the cost Hamiltonian.
        return qml.expval(cost_h)

    return circuit


def create_sampling_circuit(graph: nx.Graph, cost_h, mixer_h):
    """
    Create a quantum circuit (QNode) that returns the probability distribution of all basis states.
    
    This circuit is used *after* optimization to identify the most probable bitstrings, 
    which represent our best guesses for the optimal Max-Cut partition.

    Args:
        graph (nx.Graph): The graph representing the Max-Cut instance.
        cost_h: The cost Hamiltonian.
        mixer_h: The mixer Hamiltonian.
    
    Returns:
        function: A PennyLane QNode that takes optimized parameters and returns a probability array.
    """
    n_wires = len(graph.nodes)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        # Prepare the equal superposition state
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        # Apply the optimized QAOA layers
        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        # Instead of expectation value, return the probability of measuring each possible bitstring
        return qml.probs(wires=range(n_wires))

    return circuit
