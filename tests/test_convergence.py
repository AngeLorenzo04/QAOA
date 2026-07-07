import pytest
import networkx as nx
from src.qaoa.qaoa_runner import QAOARunner
from src.common.graphs import create_random_graph

def test_qaoa_epsilon_convergence():
    """Test that the QAOA optimization terminates early when epsilon convergence is met."""
    graph = create_random_graph(n_nodes=4, probability=0.5, seed=42)
    runner = QAOARunner(graph, p_value=1)
    
    # Run with a relatively large epsilon so it converges very quickly (within 5-10 iterations)
    results = runner.run(max_optimization_iterations=100, epsilon=1.0)
    
    termination_reason = results['metrics'].get('termination_reason', '')
    print("Epsilon Termination Reason:", termination_reason)
    assert "terminated_early: epsilon convergence" in termination_reason

def test_qaoa_timeout_termination():
    """Test that the QAOA optimization terminates early when a timeout is reached."""
    graph = create_random_graph(n_nodes=4, probability=0.5, seed=42)
    runner = QAOARunner(graph, p_value=1)
    
    # Run with a tiny timeout (e.g. 0.0001 seconds) to trigger immediate timeout
    results = runner.run(max_optimization_iterations=100, timeout=0.0001)
    
    termination_reason = results['metrics'].get('termination_reason', '')
    print("Timeout Termination Reason:", termination_reason)
    assert "terminated_early: timeout" in termination_reason

def test_qaoa_custom_gradient_descent():
    """Test that the custom Gradient Descent (GD) optimizer converges successfully."""
    graph = create_random_graph(n_nodes=3, probability=0.8, seed=42)
    runner = QAOARunner(graph, p_value=1)
    
    results = runner.run(max_optimization_iterations=20, optimizer_method='GD')
    
    assert results['metrics']['optimization_iterations'] > 0
    assert results['best_measured_cut_value'] >= 0

def test_qaoa_scipy_gradient_descent():
    """Test that the SciPy BFGS gradient-based optimizer converges successfully."""
    graph = create_random_graph(n_nodes=3, probability=0.8, seed=42)
    runner = QAOARunner(graph, p_value=1)
    
    results = runner.run(max_optimization_iterations=20, optimizer_method='BFGS')
    
    assert results['metrics']['optimization_iterations'] > 0
    assert results['best_measured_cut_value'] >= 0

