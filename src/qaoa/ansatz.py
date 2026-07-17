import networkx as nx
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import I, X, Z, PauliSumOp

from src.qaoa.mixers import get_mixer

def get_cost_hamiltonian(graph: nx.Graph) -> PauliSumOp:
    """
    Constructs the Cost Hamiltonian (H_C) for the MaxCut problem.
    H_C = sum_{<i,j> in E} (I - Z_i Z_j) / 2

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        PauliSumOp: The Cost Hamiltonian.
    """
    num_qubits = graph.number_of_nodes()
    
    h_c = 0
    for i, j in graph.edges():
        zi_zj_label = ['I'] * num_qubits
        zi_zj_label[i] = 'Z'
        zi_zj_label[j] = 'Z'
        term_zi_zj = PauliSumOp(SparsePauliOp("".join(zi_zj_label)))
        h_c += (PauliSumOp(SparsePauliOp('I' * num_qubits)) - term_zi_zj) * 0.5

    return h_c


def get_mixer_hamiltonian(graph: nx.Graph) -> PauliSumOp:
    """
    Constructs the Mixer Hamiltonian (H_M) for the MaxCut problem (standard mixer).
    H_M = sum_{i in V} X_i

    Args:
        graph (nx.Graph): The input graph (used to determine number of qubits).

    Returns:
        PauliSumOp: The Mixer Hamiltonian.
    """
    num_qubits = graph.number_of_nodes()
    h_m = 0
    for i in range(num_qubits):
        xi_label = ['I'] * num_qubits
        xi_label[i] = 'X'
        h_m += PauliSumOp(SparsePauliOp("".join(xi_label)))
    return h_m


def get_qaoa_ansatz(graph: nx.Graph, p: int, mixer: str = "standard") -> QuantumCircuit:
    """
    Constructs the QAOA ansatz circuit.

    Args:
        graph (nx.Graph): The input graph.
        p (int): The number of QAOA layers.
        mixer (str): The mixer strategy to use ('standard' or 'grover').

    Returns:
        QuantumCircuit: The QAOA ansatz circuit.
    """
    num_qubits = graph.number_of_nodes()
    qc_correct = QuantumCircuit(num_qubits)

    # Get the mixer strategy
    mixer_strategy = get_mixer(mixer)

    # Apply initial state
    mixer_strategy.apply_initial_state(qc_correct, num_qubits)

    # Define parameters for Cost and Mixer Hamiltonians
    gamma_correct = ParameterVector('gamma', p)
    beta_correct = ParameterVector('beta', p)

    for layer in range(p):
        # Cost Layer: Apply RZZ gates for each edge
        for i, j in graph.edges():
            qc_correct.rzz(2 * gamma_correct[layer], i, j)
        
        # Mixer Layer: Apply mixer unitary
        mixer_strategy.apply_mixer(qc_correct, beta_correct[layer], num_qubits)
    
    return qc_correct

if __name__ == '__main__':
    # Example usage:
    # Create a simple graph (e.g., a triangle)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    num_qubits = G.number_of_nodes()
    p_value = 1

    print(f"Graph with {num_qubits} nodes and {G.number_of_edges()} edges:")
    print(f"Edges: {G.edges()}")

    # Test Cost Hamiltonian
    h_c_op = get_cost_hamiltonian(G)
    print("\nCost Hamiltonian (H_C):")
    print(h_c_op)

    # Test Mixer Hamiltonian
    h_m_op = get_mixer_hamiltonian(G)
    print("\nStandard Mixer Hamiltonian (H_M):")
    print(h_m_op)

    # Test QAOA Ansatz Circuit with standard mixer
    qaoa_circuit = get_qaoa_ansatz(G, p_value, mixer="standard")
    print(f"\nQAOA Ansatz Circuit (Standard Mixer, p={p_value}):")
    print(qaoa_circuit.draw(output='text'))

    # Test QAOA Ansatz Circuit with Grover mixer
    qaoa_circuit_grover = get_qaoa_ansatz(G, p_value, mixer="grover")
    print(f"\nQAOA Ansatz Circuit (Grover Mixer, p={p_value}):")
    print(qaoa_circuit_grover.draw(output='text'))

