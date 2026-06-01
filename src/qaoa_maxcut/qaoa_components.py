import pennylane as qml
import networkx as nx


def build_maxcut_hamiltonians(graph: nx.Graph):
    """Return the cost and mixer Hamiltonians for a Max-Cut instance."""
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    return cost_h, mixer_h