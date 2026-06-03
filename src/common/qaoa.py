import pennylane as qml

def apply_qaoa_layer(gamma: float, beta: float, cost_h, mixer_h) -> None:
    """
    Apply a single QAOA layer (cost evolution followed by mixer evolution).
    """
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)

def create_qaoa_circuit(n_wires: int, cost_h, mixer_h):
    """
    Create a QNode for QAOA optimization.
    """
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        return qml.expval(cost_h)

    return circuit

def create_sampling_circuit(n_wires: int, cost_h, mixer_h):
    """
    Create a QNode for sampling (returning probabilities).
    """
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(params):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        gammas = params[0]
        betas = params[1]
        for gamma, beta in zip(gammas, betas):
            apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        return qml.probs(wires=range(n_wires))

    return circuit
