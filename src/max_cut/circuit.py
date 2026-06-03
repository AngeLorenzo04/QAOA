from common.qaoa import create_qaoa_circuit as _create_qaoa_circuit
from common.qaoa import create_sampling_circuit as _create_sampling_circuit

def create_maxcut_circuit(graph, cost_h, mixer_h):
    return _create_qaoa_circuit(len(graph.nodes), cost_h, mixer_h)

def create_maxcut_sampling_circuit(graph, cost_h, mixer_h):
    return _create_sampling_circuit(len(graph.nodes), cost_h, mixer_h)
