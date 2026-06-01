import pennylane as qml
import networkx as nx


def apply_qaoa_layer(gamma: float, beta: float, cost_h, mixer_h) -> None:
    """Apply one QAOA layer."""
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)


def create_qaoa_circuit(graph: nx.Graph, cost_h, mixer_h):
    """Create a QNode that evaluates the expectation value of the cost Hamiltonian."""
    n_wires = len(graph.nodes)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(gamma: float, beta: float):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)

        apply_qaoa_layer(gamma, beta, cost_h, mixer_h)

        return qml.expval(cost_h)

    return circuit