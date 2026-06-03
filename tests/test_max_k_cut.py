import pytest
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

from max_k_cut.components import build_max_k_cut_hamiltonians
from max_k_cut.circuit import create_max_k_cut_circuit
from max_k_cut.utils import create_complete_graph

def test_max_k_cut_hamiltonian_construction():
    """Testa la costruzione dell'Hamiltoniana Max-k-Cut con termini di penalità."""
    k = 3
    graph = nx.Graph([(0, 1)])
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k, penalty_weight=1.0)
    
    assert isinstance(cost_h, qml.Hamiltonian)
    # n_nodes * k qubits = 2 * 3 = 6
    assert len(cost_h.wires) == 6

def test_max_k_cut_circuit_execution():
    """Testa l'esecuzione del circuito Max-k-Cut."""
    k = 2
    n_nodes = 3
    graph = create_complete_graph(n_nodes)
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k)
    circuit = create_max_k_cut_circuit(graph, k, cost_h, mixer_h)
    
    params = np.array([[0.1], [0.1]], requires_grad=True)
    val = circuit(params)
    
    assert isinstance(val, (float, np.tensor))

def test_max_k_cut_different_k():
    """Verifica che l'algoritmo accetti diversi valori di k."""
    for k in [2, 3, 4]:
        graph = nx.Graph([(0, 1)])
        cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k)
        assert len(cost_h.wires) == 2 * k

def test_max_k_cut_optimization_step():
    """Testa se l'ottimizzatore Adam funziona con Max-k-Cut."""
    k = 2
    graph = nx.Graph([(0, 1)])
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k)
    circuit = create_max_k_cut_circuit(graph, k, cost_h, mixer_h)
    
    params = np.array([[0.1], [0.1]], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    new_params = opt.step(circuit, params)
    assert not np.allclose(params, new_params)
