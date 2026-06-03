import pytest
import networkx as nx
from max_cut.utils import create_cycle_graph
from max_k_cut.utils import create_complete_graph

def test_cycle_graph_properties():
    """Verifica le proprietà del grafo a ciclo."""
    n = 5
    graph = create_cycle_graph() # Default is 4
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
    
    # Verifica che sia un ciclo (ogni nodo ha grado 2)
    for node in graph.nodes:
        assert graph.degree(node) == 2

def test_complete_graph_properties():
    """Verifica le proprietà del grafo completo."""
    n = 4
    graph = create_complete_graph(n)
    assert len(graph.nodes) == n
    # Numero di archi in una clique: n*(n-1)/2
    assert len(graph.edges) == n * (n - 1) // 2
    
    # Ogni nodo deve essere connesso a tutti gli altri
    for node in graph.nodes:
        assert graph.degree(node) == n - 1
