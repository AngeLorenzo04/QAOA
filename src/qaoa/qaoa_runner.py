import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler, Estimator
from qiskit.opflow import PauliSumOp
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.qaoa.ansatz import get_qaoa_ansatz, get_cost_hamiltonian
from src.qaoa.optimizer import qaoa_optimizer, calculate_maxcut_value # Import calculate_maxcut_value for final evaluation

class QAOARunner:
    """
    A class to encapsulate the execution of QAOA for a single graph
    with a given configuration.
    """
    def __init__(self, graph: nx.Graph, p_value: int, mixer_type: str = "standard", encoding_type: str = "binary"):
        # Validate that the graph is valid (no isolated nodes)
        from src.common.graphs import is_valid_graph
        if not is_valid_graph(graph):
            raise ValueError("Il grafo fornito non è valido: ogni nodo deve essere connesso ad almeno un altro nodo (nessun nodo isolato).")

        self.graph = graph
        self.p_value = p_value
        self.mixer_type = mixer_type # Currently only "standard" is supported
        self.encoding_type = encoding_type # Currently only "binary" is supported
        self.num_qubits = graph.number_of_nodes()

        # Initialize Qiskit primitives
        self.sampler = Sampler()
        self.estimator = Estimator()

        # Build QAOA components
        self.ansatz_circuit = get_qaoa_ansatz(self.graph, self.p_value)
        self.cost_hamiltonian = get_cost_hamiltonian(self.graph)

    def run(
        self,
        max_optimization_iterations: int = 100,
        shots: int = 1024,
        optimizer_method: str = 'COBYLA',
        tol: Optional[float] = None,
        epsilon: Optional[float] = None,
        timeout: Optional[float] = None,
        verbose: bool = False,
        learning_rate: float = 0.1,
        dx: float = 0.2,
        gd_method: str = 'adam',
        num_starts: int = 1
    ) -> Dict[str, Any]:
        """
        Runs the QAOA optimization and simulation for the configured graph and parameters.

        Args:
            max_optimization_iterations (int): Maximum iterations for the classical optimizer.
            shots (int): Number of shots for circuit sampling.
            optimizer_method (str): Classical optimization method (e.g., 'COBYLA', 'SLSQP', 'GD').
            tol (Optional[float]): Tolerance for termination passed to SciPy minimize.
            epsilon (Optional[float]): Epsilon convergence threshold for custom early stopping.
            timeout (Optional[float]): Timeout threshold in seconds for early stopping.
            verbose (bool): Whether to print run details to stdout.
            learning_rate (float): Learning rate for custom gradient descent.
            dx (float): Finite difference step size for gradient estimation.
            gd_method (str): Custom GD variant to run ('vanilla', 'momentum', 'adam').
            num_starts (int): Number of random initial starting points for multi-start GD optimization.

        Returns:
            Dict[str, Any]: A dictionary containing QAOA results and metrics.
        """
        if verbose:
            print(f"  Running QAOA for N={self.num_qubits}, p={self.p_value}, mixer={self.mixer_type}, encoding={self.encoding_type} (optimizer={optimizer_method})")

        # 1. Initialize parameters
        # Qiskit parameters are 2*p_value: p gammas and p betas
        initial_params = np.random.rand(2 * self.p_value) * 2 * np.pi # gamma and beta between 0 and 2pi

        # 2. Optimize QAOA parameters
        optimization_results = qaoa_optimizer(
            self.ansatz_circuit,
            self.graph,
            self.cost_hamiltonian,
            initial_params,
            sampler=self.sampler,
            optimizer_method=optimizer_method,
            max_iterations=max_optimization_iterations,
            tol=tol,
            epsilon=epsilon,
            timeout=timeout,
            learning_rate=learning_rate,
            dx=dx,
            gd_method=gd_method,
            num_starts=num_starts
        )
        optimal_params = optimization_results['optimal_params']
        num_optimization_iterations = optimization_results['num_iterations']
        termination_reason = optimization_results.get('termination_reason', 'optimizer_completed')
        optimization_history = optimization_results['history'] # Extract history
        trajectory_params = optimization_results.get('trajectory_params', [])
        
        # 3. Bind optimal parameters to the ansatz circuit
        optimal_circuit = self.ansatz_circuit.assign_parameters(dict(zip(self.ansatz_circuit.parameters, optimal_params)))

        # 4. Run the optimal circuit on a sampler to get measurement outcomes
        measured_optimal_circuit = optimal_circuit.measure_all(inplace=False)
        job = self.sampler.run(measured_optimal_circuit, shots=shots)
        quasi_distribution = job.result().quasi_dists[0]

        # 5. Calculate QAOA MaxCut value (expected value) and best measured cut
        qaoa_expected_cut_value = 0.0
        best_measured_cut_value = -1
        best_measured_bitstring = ""

        # Find the most probable bitstring(s) and their cut values
        sorted_outcomes = sorted(quasi_distribution.items(), key=lambda item: item[1], reverse=True)
        
        for state_int, prob in sorted_outcomes:
            bitstring = format(state_int, f'0{self.num_qubits}b')[::-1]
            cut_val = calculate_maxcut_value(self.graph, bitstring)
            qaoa_expected_cut_value += prob * cut_val
            
            if cut_val > best_measured_cut_value:
                best_measured_cut_value = cut_val
                best_measured_bitstring = bitstring
            # If there are multiple bitstrings with the same max cut value, we just pick the first one

        # Calculate circuit depth and number of parameters
        circuit_depth = optimal_circuit.depth()
        num_parameters = len(optimal_circuit.parameters) # Should be 2 * p_value

        return {
            'optimal_params': optimal_params,
            'qaoa_expected_cut_value': qaoa_expected_cut_value,
            'best_measured_cut_value': best_measured_cut_value,
            'best_measured_bitstring': best_measured_bitstring,
            'quasi_distribution': {format(k, f'0{self.num_qubits}b')[::-1]: float(v) for k, v in quasi_distribution.items()}, # Convert keys to bitstrings and prob to float for JSON
            'metrics': {
                'circuit_depth': circuit_depth,
                'num_parameters': num_parameters,
                'optimization_iterations': num_optimization_iterations,
                'termination_reason': termination_reason,
                'optimization_history': optimization_history, # Include optimization history
                'trajectory_params': trajectory_params, # Include trajectory parameters
                'all_trajectories': optimization_results.get('all_trajectories', [trajectory_params]), # Include all trajectories for multi-start
                'total_shots': shots # Assuming shots are constant for sampling
            }
        }

if __name__ == '__main__':
    # Example usage:
    # Create a simple graph (e.g., a triangle)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    p_value = 1

    runner = QAOARunner(G, p_value)
    qaoa_results = runner.run()

    print("\nQAOA Runner Results:")
    print(f"Optimal Parameters: {qaoa_results['optimal_params']}")
    print(f"QAOA Expected Cut Value: {qaoa_results['qaoa_expected_cut_value']:.4f}")
    print(f"Best Measured Cut Value: {qaoa_results['best_measured_cut_value']}")
    print(f"Best Measured Bitstring: {qaoa_results['best_measured_bitstring']}")
    print(f"Circuit Depth: {qaoa_results['metrics']['circuit_depth']}")
    print(f"Number of Parameters: {qaoa_results['metrics']['num_parameters']}")
    print(f"Optimization Iterations: {qaoa_results['metrics']['optimization_iterations']}")
    
    # Compare with exact MaxCut
    from src.data.exact_maxcut_solver import find_exact_maxcut
    exact_maxcut_result = find_exact_maxcut(G)
    print(f"\nExact MaxCut Value: {exact_maxcut_result['max_cut_value']}")
    
    approximation_ratio = qaoa_results['best_measured_cut_value'] / exact_maxcut_result['max_cut_value']
    print(f"Approximation Ratio (based on best measured): {approximation_ratio:.4f}")
