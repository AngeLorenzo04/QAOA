import networkx as nx
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.result import QuasiDistribution
from qiskit.circuit import QuantumCircuit, ParameterVector
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple, Callable
import numpy as np

# Assuming get_cost_hamiltonian is available from ansatz module for expectation value calculation
# In a real scenario, this would be passed or imported from a higher level.
# For now, let's just re-declare it or import it from ansatz.
from src.qaoa.ansatz import get_cost_hamiltonian

def calculate_maxcut_value(graph: nx.Graph, bitstring: str) -> int:
    """
    Calculates the cut value for a given graph and bitstring assignment.
    The bitstring represents the partition, where '0' means node in set 0, '1' in set 1.
    """
    cut_value = 0
    partition = [int(b) for b in bitstring]
    for u, v in graph.edges():
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value

def get_objective_function(
    ansatz_circuit: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    graph: nx.Graph,
    sampler: StatevectorSampler
) -> Callable[[np.ndarray], float]:
    """
    Returns the objective function to be minimized by the classical optimizer.
    This function calculates the expectation value of the cost Hamiltonian.

    Args:
        ansatz_circuit (QuantumCircuit): The QAOA ansatz circuit with unbound parameters.
        cost_hamiltonian (PauliSumOp): The cost Hamiltonian for the MaxCut problem.
        graph (nx.Graph): The graph for which to calculate the cut value (needed for sampling method).
        sampler (Sampler): Qiskit Sampler primitive for expectation value calculation.

    Returns:
        Callable[[np.ndarray], float]: An objective function that takes a flat array of
                                        parameters (gamma and beta) and returns the
                                        negative of the expectation value (since we minimize).
    """
    num_qubits = ansatz_circuit.num_qubits
    
    # Store history of evaluations
    history = {'fun': [], 'params': []}

    def objective_function(params: np.ndarray) -> float:
        # Split params into gamma and beta
        p = len(ansatz_circuit.parameters) // 2
        gamma_params = params[:p]
        beta_params = params[p:]
        
        # Bind parameters to the ansatz circuit
        bound_circuit = ansatz_circuit.bind_parameters(dict(zip(ansatz_circuit.parameters, params)))

        # Calculate expectation value using the sampler
        # The sampler primitive directly calculates expectation values for SparsePauliOp
        # by passing the operator with the circuit.
        # Note: cost_hamiltonian needs to be a Qiskit Operator (e.g., SparsePauliOp)
        # for direct expectation value calculation with primitives.
        
        # If cost_hamiltonian is PauliSumOp, convert to SparsePauliOp for sampler
        # This conversion might be implicit in future Qiskit versions or needs a specific method.
        # For now, let's assume it can be directly used, or we manually calculate from counts.
        
        # Option 1: Direct expectation value via Sampler (preferred if supported directly by PauliSumOp)
        job = sampler.run(bound_circuit, shots=1024) # Increased shots for better estimation
        # Assuming sampler.run can handle PauliSumOp or it's implicitly converted
        # For current Qiskit Primitives, we might need to convert PauliSumOp to Estimator's expectation value context.
        # Let's adjust this to manually calculate expectation from quasi-distributions.
        
        # Option 2: Calculate expectation from quasi-distributions (more general)
        job = sampler.run(bound_circuit, shots=1024)
        quasi_distribution: QuasiDistribution = job.result().quasi_dists[0]

        exp_val = 0.0
        for bitstring, prob in quasi_distribution.items():
            # For MaxCut, the value is higher for better cuts.
            # The cost Hamiltonian in Qiskit is usually defined such that
            # its expectation value is proportional to the negative of the cut value (to minimize).
            # H_C = sum_{<i,j>} (1 - Z_i Z_j) / 2
            # For a Z_i Z_j term, if i and j are in different sets, Z_i Z_j = -1, (1 - (-1))/2 = 1.
            # If i and j are in same set, Z_i Z_j = 1, (1 - 1)/2 = 0.
            # So, the expectation value of H_C is the cut value. We want to maximize the cut value,
            # hence we minimize -E[H_C].

            # cost_hamiltonian.expectation_value(Statevector.from_int(int(bitstring, 2), num_qubits))
            # The PauliSumOp.expectation_value method takes a Statevector or an Operator.
            # Instead of manually calculating from bitstrings, let's assume `sampler` directly gives expectation value
            # if we pass the operator to its `run` method, which is the intended usage of `Estimator` primitive.
            # The `Sampler` primitive gives measurement outcomes.
            
            # Let's use `get_cost_hamiltonian` to define the cost for a given bitstring directly.
            # We need the `calculate_maxcut_value` to convert bitstring to cut value.
            maxcut_val_for_bitstring = calculate_maxcut_value(graph, bitstring)
            # The cost Hamiltonian is defined such that its expectation value *is* the cut value
            # E[H_C] = sum_x P(x) * <x|H_C|x> = sum_x P(x) * (cut_value(x))
            # We want to maximize this, so we minimize its negative.
            exp_val += prob * maxcut_val_for_bitstring
            
        neg_expectation_value = -exp_val

        history['fun'].append(neg_expectation_value)
        history['params'].append(params.tolist())

        return neg_expectation_value

    return objective_function, history


def qaoa_optimizer(
    ansatz_circuit: QuantumCircuit,
    graph: nx.Graph,
    cost_hamiltonian: SparsePauliOp,
    initial_params: np.ndarray,
    optimizer_method: str = 'COBYLA',
    max_iterations: int = 100,
    sampler: StatevectorSampler = None
) -> Dict[str, Any]:
    """
    Performs classical optimization to find optimal QAOA parameters.

    Args:
        ansatz_circuit (QuantumCircuit): The QAOA ansatz circuit with unbound parameters.
        graph (nx.Graph): The graph associated with the MaxCut problem.
        cost_hamiltonian (PauliSumOp): The cost Hamiltonian for the MaxCut problem.
        initial_params (np.ndarray): Initial guess for the QAOA parameters (gamma and beta).
        optimizer_method (str): Classical optimization method (e.g., 'COBYLA', 'SLSQP').
        max_iterations (int): Maximum number of classical optimization iterations.
        sampler (Sampler): Qiskit Sampler primitive instance.

    Returns:
        Dict[str, Any]: A dictionary containing optimization results:
                        - 'optimal_params': Optimized gamma and beta values.
                        - 'optimal_value': The minimized objective function value.
                        - 'num_iterations': Number of classical optimization iterations.
                        - 'history': A dictionary containing the optimization history.
    """
    if sampler is None:
        sampler = StatevectorSampler() # Use default sampler if not provided

    objective_function, history = get_objective_function(ansatz_circuit, cost_hamiltonian, graph, sampler)

    # Perform classical optimization
    result = minimize(
        objective_function,
        initial_params,
        method=optimizer_method,
        options={'maxiter': max_iterations, 'disp': False} # disp=True for verbose output
    )

    optimal_params = result.x
    optimal_value = result.fun
    num_iterations = result.nit

    return {
        'optimal_params': optimal_params.tolist(),
        'optimal_value': optimal_value,
        'num_iterations': num_iterations,
        'history': history # Store the history of objective function values and parameters
    }

if __name__ == '__main__':
    # Example usage:
    # Create a simple graph (e.g., a triangle)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    num_qubits = G.number_of_nodes()
    p_value = 1

    # Get the ansatz circuit and cost Hamiltonian
    from src.qaoa.ansatz import get_qaoa_ansatz
    ansatz_circuit = get_qaoa_ansatz(G, p_value)
    cost_hamiltonian = get_cost_hamiltonian(G)

    # Initialize parameters randomly
    initial_params = np.random.rand(2 * p_value) * 2 * np.pi # gamma and beta between 0 and 2pi

    # Instantiate Qiskit Sampler
    sampler = StatevectorSampler()

    print(f"Starting QAOA optimization for N={num_qubits}, p={p_value}...")
    optimization_results = qaoa_optimizer(
        ansatz_circuit,
        G,
        cost_hamiltonian,
        initial_params,
        sampler=sampler
    )

    print("\nOptimization Results:")
    print(f"Optimal Parameters: {optimization_results['optimal_params']}")
    print(f"Optimal (Negative) Expectation Value: {optimization_results['optimal_value']}")
    print(f"Number of Iterations: {optimization_results['num_iterations']}")
    print(f"Optimization History Length: {len(optimization_results['history']['fun'])}")

    # To get the actual max cut value from the optimal_value (which is -E[H_C])
    print(f"MaxCut Expectation Value: {-optimization_results['optimal_value']}")

    # After optimization, you can bind the optimal parameters and sample the circuit
    optimal_circuit = ansatz_circuit.bind_parameters(optimization_results['optimal_params'])
    job = sampler.run(optimal_circuit, shots=1024)
    quasi_distribution = job.result().quasi_dists[0]

    print("\nTop 5 Measurement Outcomes:")
    sorted_outcomes = sorted(quasi_distribution.items(), key=lambda item: item[1], reverse=True)
    for bitstring, prob in sorted_outcomes[:5]:
        cut_val = calculate_maxcut_value(G, bitstring)
        print(f"  Bitstring: {bitstring}, Probability: {prob:.4f}, Cut Value: {cut_val}")

    # Check if the solution matches the exact MaxCut
    from src.data.exact_maxcut_solver import find_exact_maxcut
    exact_maxcut_result = find_exact_maxcut(G)
    print(f"\nExact MaxCut Value: {exact_maxcut_result['max_cut_value']}")
