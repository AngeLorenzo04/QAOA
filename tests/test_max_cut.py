import pytest
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

from max_cut.components import build_maxcut_hamiltonians
from max_cut.circuit import create_maxcut_circuit
from common.graphs import create_cycle_graph

def test_max_cut_hamiltonian_construction():
    """Testa se le Hamiltoniane vengono costruite correttamente per un grafo semplice."""
    graph = nx.Graph([(0, 1)])
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)
    
    assert isinstance(cost_h, qml.Hamiltonian)
    assert isinstance(mixer_h, qml.Hamiltonian)
    # Per Max-Cut su 2 nodi, l'Hamiltoniana di costo ha termini Z
    cost_str = str(cost_h)
    assert "PauliZ" in cost_str or "Z(" in cost_str

def test_max_cut_circuit_execution():
    """Testa se il circuito QAOA può essere eseguito senza errori."""
    graph = create_cycle_graph(4)
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)
    circuit = create_maxcut_circuit(graph, cost_h, mixer_h)
    
    p = 1
    params = np.array([np.random.rand(p), np.random.rand(p)], requires_grad=True)
    
    val = circuit(params)
    assert isinstance(val, (float, np.tensor))

def test_max_cut_optimization_step():
    """Testa se un passo di ottimizzazione funziona (i gradienti vengono calcolati)."""
    graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)
    circuit = create_maxcut_circuit(graph, cost_h, mixer_h)
    
    params = np.array([[0.1], [0.1]], requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=0.1)
    
    new_params = opt.step(circuit, params)
    
    assert not np.allclose(params, new_params)
