import networkx as nx
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
import time

class OptimizationTerminator(Exception):
    """Custom exception to stop SciPy optimization early on convergence or timeout."""
    def __init__(self, params: np.ndarray, value: float, reason: str):
        super().__init__(reason)
        self.params = params
        self.value = value
        self.reason = reason

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
    cost_hamiltonian: PauliSumOp,
    graph: nx.Graph,
    sampler: Sampler,
    epsilon: Optional[float] = None,
    timeout: Optional[float] = None
) -> Tuple[Callable[[np.ndarray], float], Dict[str, Any]]:
    """
    Returns the objective function to be minimized by the classical optimizer.
    This function calculates the expectation value of the cost Hamiltonian.

    Args:
        ansatz_circuit (QuantumCircuit): The QAOA ansatz circuit with unbound parameters.
        cost_hamiltonian (PauliSumOp): The cost Hamiltonian for the MaxCut problem.
        graph (nx.Graph): The graph for which to calculate the cut value (needed for sampling method).
        sampler (Sampler): Qiskit Sampler primitive for expectation value calculation.
        epsilon (Optional[float]): Convergence threshold. Stop if objective function value variation
                                   over the last 5 iterations is less than epsilon.
        timeout (Optional[float]): Timeout threshold in seconds. Stop if optimization exceeds this duration.

    Returns:
        Tuple[Callable[[np.ndarray], float], Dict[str, Any]]:
            - An objective function that takes a flat array of parameters (gamma and beta)
              and returns the negative of the expectation value.
            - A dictionary containing the optimization history.
    """
    num_qubits = ansatz_circuit.num_qubits
    
    # Store history of evaluations
    history = {'fun': [], 'params': []}
    
    # Track best params, value, and start time for early termination recovery
    best_value = float('inf')
    best_params = None
    start_time = time.time()

    def objective_function(params: np.ndarray) -> float:
        nonlocal best_value, best_params
        
        # Check timeout first
        if timeout is not None and (time.time() - start_time) > timeout:
            raise OptimizationTerminator(
                best_params if best_params is not None else params,
                best_value if best_value != float('inf') else 0.0,
                f"timeout ({timeout}s exceeded)"
            )

        # Split params into beta and gamma (since beta is first alphabetically: [beta, gamma])
        p = len(ansatz_circuit.parameters) // 2
        beta_params = params[:p]
        gamma_params = params[p:]
        
        # Bind parameters to the ansatz circuit by their explicit name
        param_dict = {}
        for param in ansatz_circuit.parameters:
            name = param.name
            if 'beta' in name:
                idx = int(name.split('[')[1].split(']')[0])
                param_dict[param] = beta_params[idx]
            elif 'gamma' in name:
                idx = int(name.split('[')[1].split(']')[0])
                param_dict[param] = gamma_params[idx]
                
        bound_circuit = ansatz_circuit.assign_parameters(param_dict)

        # Ensure the circuit has measurements before running the sampler
        measured_circuit = bound_circuit.measure_all(inplace=False)
        
        # Calculate expectation from quasi-distributions (more general)
        job = sampler.run(measured_circuit, shots=1024)
        quasi_distribution: QuasiDistribution = job.result().quasi_dists[0]

        exp_val = 0.0
        for state_int, prob in quasi_distribution.items():
            # Convert integer state to bitstring
            bitstring = format(state_int, f'0{num_qubits}b')
            maxcut_val_for_bitstring = calculate_maxcut_value(graph, bitstring)
            exp_val += prob * maxcut_val_for_bitstring
            
        neg_expectation_value = -exp_val

        history['fun'].append(neg_expectation_value)
        history['params'].append(params.tolist())

        if neg_expectation_value < best_value:
            best_value = neg_expectation_value
            best_params = params.copy()

        # Check custom epsilon convergence (range of last 5 evaluations)
        if epsilon is not None and len(history['fun']) >= 5:
            last_vals = history['fun'][-5:]
            val_range = max(last_vals) - min(last_vals)
            if val_range < epsilon:
                raise OptimizationTerminator(
                    best_params,
                    best_value,
                    f"epsilon convergence (< {epsilon})"
                )

        return neg_expectation_value

    return objective_function, history

def custom_gradient_descent(
    objective_function: Callable[[np.ndarray], float],
    x0: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    tol: float = 1e-5,
    dx: float = 0.2,
    method: str = 'adam'
) -> Tuple[np.ndarray, float, int, str, List[List[float]]]:
    """
    Custom Gradient Descent optimizer using central finite differences for gradient estimation.
    Supports 'vanilla' gradient descent, momentum, and 'adam'.
    Uses modulo wrapping to keep parameters within [0, 2*pi] (gamma) and [0, pi] (beta).
    """
    x = np.array(x0, dtype=float).copy()
    n = len(x)
    p = n // 2
    
    # Wrap initial parameters (x[:p] is beta, x[p:] is gamma)
    x[:p] = np.mod(x[:p], np.pi)     # beta in [0, pi]
    x[p:] = np.mod(x[p:], 2 * np.pi) # gamma in [0, 2*pi]
    
    trajectory_params = [x.copy().tolist()]
    num_iterations = 0
    termination_reason = "max_iterations_reached"

    # Momentum / Adam state variables
    m = np.zeros(n)
    v_adam = np.zeros(n)
    v_momentum = np.zeros(n)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8

    for i in range(max_iterations):
        num_iterations = i + 1
        
        # Evaluate current function value
        f_val = objective_function(x)
        
        # Estimate gradient using central differences
        grad = np.zeros(n)
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += dx
            x_minus = x.copy()
            x_minus[j] -= dx
            
            f_plus = objective_function(x_plus)
            f_minus = objective_function(x_minus)
            grad[j] = (f_plus - f_minus) / (2 * dx)
            
        # Check gradient norm convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            termination_reason = f"converged: gradient norm ({grad_norm:.6f}) < tol ({tol})"
            break
            
        # Update position using specified method
        if method.lower() == 'adam':
            m = beta1 * m + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** num_iterations)
            v_hat = v_adam / (1 - beta2 ** num_iterations)
            x -= learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)
        elif method.lower() == 'momentum':
            v_momentum = 0.9 * v_momentum + learning_rate * grad
            x -= v_momentum
        else: # vanilla gradient descent
            x -= learning_rate * grad
        
        # Keep parameters within periodic boundaries
        x[:p] = np.mod(x[:p], np.pi)     # beta in [0, pi]
        x[p:] = np.mod(x[p:], 2 * np.pi) # gamma in [0, 2*pi]
        
        trajectory_params.append(x.copy().tolist())

    # Final evaluation
    f_final = objective_function(x)
    return x, f_final, num_iterations, termination_reason, trajectory_params


def qaoa_optimizer(
    ansatz_circuit: QuantumCircuit,
    graph: nx.Graph,
    cost_hamiltonian: PauliSumOp,
    initial_params: np.ndarray,
    optimizer_method: str = 'COBYLA',
    max_iterations: int = 100,
    sampler: Sampler = None,
    tol: Optional[float] = None,
    epsilon: Optional[float] = None,
    timeout: Optional[float] = None,
    learning_rate: float = 0.1,
    dx: float = 0.2,
    gd_method: str = 'adam',
    num_starts: int = 1
) -> Dict[str, Any]:
    """
    Performs classical optimization to find optimal QAOA parameters.

    Args:
        ansatz_circuit (QuantumCircuit): The QAOA ansatz circuit with unbound parameters.
        graph (nx.Graph): The graph associated with the MaxCut problem.
        cost_hamiltonian (PauliSumOp): The cost Hamiltonian for the MaxCut problem.
        initial_params (np.ndarray): Initial guess for the QAOA parameters (gamma and beta).
        optimizer_method (str): Classical optimization method (e.g., 'COBYLA', 'SLSQP', 'GD').
        max_iterations (int): Maximum number of classical optimization iterations.
        sampler (Sampler): Qiskit Sampler primitive instance.
        tol (Optional[float]): Tolerance for termination passed to SciPy minimize.
        epsilon (Optional[float]): Epsilon convergence threshold for custom early stopping.
        timeout (Optional[float]): Timeout threshold in seconds for early stopping.
        learning_rate (float): Learning rate for custom gradient descent.
        dx (float): Finite difference step size for gradient estimation.
        gd_method (str): Custom GD variant to run ('vanilla', 'momentum', 'adam').
        num_starts (int): Number of random initial starting points for multi-start GD optimization.

    Returns:
        Dict[str, Any]: A dictionary containing optimization results:
                        - 'optimal_params': Optimized gamma and beta values.
                        - 'optimal_value': The minimized objective function value.
                        - 'num_iterations': Number of classical optimization iterations.
                        - 'termination_reason': Message describing how optimization ended.
                        - 'history': A dictionary containing the optimization history.
                        - 'trajectory_params': List of parameters at each optimization step.
    """
    if sampler is None:
        sampler = Sampler() # Use default sampler if not provided

    objective_function, history = get_objective_function(
        ansatz_circuit, cost_hamiltonian, graph, sampler, epsilon=epsilon, timeout=timeout
    )

    trajectory_params = [initial_params.copy().tolist()]
    termination_reason = "optimizer_completed"
    try:
        if optimizer_method.upper() == 'GD':
            best_params = None
            best_value = float('inf')
            best_num_iterations = 0
            best_termination_reason = ""
            best_trajectory_params = []
            
            # Genera i punti di partenza (il primo è quello fornito)
            starts = [initial_params.copy()]
            p = len(ansatz_circuit.parameters) // 2
            for _ in range(num_starts - 1):
                starts.append(np.random.rand(2 * p) * 2 * np.pi)
                
            for start_idx, start_p in enumerate(starts):
                opt_params, opt_value, num_iter, term_reason, traj_params = custom_gradient_descent(
                    objective_function,
                    start_p,
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    tol=tol if tol is not None else 1e-5,
                    dx=dx,
                    method=gd_method
                )
                if opt_value < best_value:
                    best_value = opt_value
                    best_params = opt_params
                    best_num_iterations = num_iter
                    best_termination_reason = f"[Run {start_idx + 1}] {term_reason}"
                    best_trajectory_params = traj_params
            
            optimal_params = best_params
            optimal_value = best_value
            num_iterations = best_num_iterations
            termination_reason = best_termination_reason
            trajectory_params = best_trajectory_params
        else:
            # Perform classical optimization using SciPy minimize
            result = minimize(
                objective_function,
                initial_params,
                method=optimizer_method,
                tol=tol,
                callback=lambda xk: trajectory_params.append(xk.copy().tolist()),
                options={'maxiter': max_iterations, 'disp': False} # disp=True for verbose output
            )
            optimal_params = result.x
            optimal_value = result.fun
            num_iterations = getattr(result, 'nit', getattr(result, 'nfev', 0))
            termination_reason = getattr(result, 'message', 'completed successfully')
    except OptimizationTerminator as e:
        optimal_params = e.params
        optimal_value = e.value
        num_iterations = len(history['fun'])
        termination_reason = f"terminated_early: {e.reason}"

    return {
        'optimal_params': optimal_params.tolist() if hasattr(optimal_params, 'tolist') else list(optimal_params),
        'optimal_value': optimal_value,
        'num_iterations': num_iterations,
        'termination_reason': termination_reason,
        'history': history, # Store the history of all objective function evaluations
        'trajectory_params': trajectory_params # Store only the main steps of parameters
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
    sampler = Sampler()

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
    optimal_circuit = ansatz_circuit.assign_parameters(dict(zip(ansatz_circuit.parameters, optimization_results['optimal_params'])))
    measured_optimal_circuit = optimal_circuit.measure_all(inplace=False)
    job = sampler.run(measured_optimal_circuit, shots=1024)
    quasi_distribution = job.result().quasi_dists[0]

    print("\nTop 5 Measurement Outcomes:")
    sorted_outcomes = sorted(quasi_distribution.items(), key=lambda item: item[1], reverse=True)
    for state_int, prob in sorted_outcomes[:5]:
        bitstring = format(state_int, f'0{num_qubits}b')
        cut_val = calculate_maxcut_value(G, bitstring)
        print(f"  Bitstring: {bitstring}, Probability: {prob:.4f}, Cut Value: {cut_val}")

    # Check if the solution matches the exact MaxCut
    from src.data.exact_maxcut_solver import find_exact_maxcut
    exact_maxcut_result = find_exact_maxcut(G)
    print(f"\nExact MaxCut Value: {exact_maxcut_result['max_cut_value']}")
