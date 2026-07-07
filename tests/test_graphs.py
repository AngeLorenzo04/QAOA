import pytest
import networkx as nx
from common.graphs import create_cycle_graph, create_complete_graph

def test_cycle_graph_properties():
    """Verifica le proprietà del grafo a ciclo."""
    graph = create_cycle_graph(4)
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

def test_random_graph_properties():
    """Verifica che il grafo casuale non abbia nodi isolati ed è riproducibile."""
    from common.graphs import create_random_graph
    
    # Verifica che venga sollevata ValueError per n_nodes < 2
    with pytest.raises(ValueError):
        create_random_graph(n_nodes=1)
        
    # Genera diversi grafi casuali con bassa densità e verifica l'assenza di nodi isolati
    for n in [3, 5, 10]:
        for prob in [0.0, 0.1, 0.5]:
            for seed in [1, 42, 100]:
                graph = create_random_graph(n_nodes=n, probability=prob, seed=seed)
                assert len(graph.nodes) == n
                for node in graph.nodes:
                    assert graph.degree(node) >= 1, f"Node {node} in graph (N={n}, P={prob}, S={seed}) is isolated!"
                    
    # Verifica la riproducibilità (stesso seed produce lo stesso grafo)
    g1 = create_random_graph(n_nodes=8, probability=0.1, seed=42)
    g2 = create_random_graph(n_nodes=8, probability=0.1, seed=42)
    assert nx.is_isomorphic(g1, g2)
    assert list(g1.edges()) == list(g2.edges())

def test_is_valid_graph():
    """Verifica il comportamento di is_valid_graph."""
    from common.graphs import is_valid_graph
    
    # Grafo vuoto -> non valido
    g_empty = nx.Graph()
    assert not is_valid_graph(g_empty)
    
    # Grafo con un solo nodo isolato -> non valido
    g_single = nx.Graph()
    g_single.add_node(0)
    assert not is_valid_graph(g_single)
    
    # Grafo con nodi isolati -> non valido
    g_isolated = nx.Graph()
    g_isolated.add_edges_from([(0, 1)])
    g_isolated.add_node(2)
    assert not is_valid_graph(g_isolated)
    
    # Grafo a ciclo (ogni nodo ha grado 2) -> valido
    g_cycle = nx.cycle_graph(4)
    assert is_valid_graph(g_cycle)
    
    # Grafo completo -> valido
    g_complete = nx.complete_graph(3)
    assert is_valid_graph(g_complete)

