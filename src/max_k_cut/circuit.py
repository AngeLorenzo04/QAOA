from common.qaoa import create_qaoa_circuit as _create_qaoa_circuit
from common.qaoa import create_sampling_circuit as _create_sampling_circuit

def create_max_k_cut_circuit(graph, k, cost_h, mixer_h):
    n_qubits = len(graph.nodes) * k
    return _create_qaoa_circuit(n_qubits, cost_h, mixer_h)

def create_max_k_cut_sampling_circuit(graph, k, cost_h, mixer_h):
    n_qubits = len(graph.nodes) * k
    return _create_sampling_circuit(n_qubits, cost_h, mixer_h)
