import pennylane as qml
import networkx as nx


def build_maxcut_hamiltonians(graph: nx.Graph):
    """
    Construct the Cost and Mixer Hamiltonians for the Max-Cut problem.

    - Cost Hamiltonian (H_C): Encodes the objective function of the problem. 
      Its ground state corresponds to the optimal Max-Cut solution.
    - Mixer Hamiltonian (H_M): Responsible for exploring the solution space 
      by driving quantum transitions between different states.

    Args:
        graph (nx.Graph): The problem graph.

    Returns:
        tuple: A tuple containing (cost_hamiltonian, mixer_hamiltonian).
    """
    # PennyLane provides a built-in function to generate these Hamiltonians 
    # directly from a NetworkX graph for the Max-Cut problem.
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    return cost_h, mixer_h
