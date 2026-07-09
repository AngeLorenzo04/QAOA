from typing import List, Tuple, Dict, Any, Set
import networkx as nx
import pulp
from tqdm import tqdm # Import tqdm for progress indication

def calculate_cut_value(graph: nx.Graph, partition: List[int]) -> int:
    """
    Calculates the cut value for a given graph and partition.

    Args:
        graph (nx.Graph): The input graph.
        partition (List[int]): A list where partition[i] = 0 or 1, indicating
                                which set node i belongs to.

    Returns:
        int: The number of edges crossing the partition.
    """
    cut_value = 0
    for u, v in graph.edges():
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value

def find_exact_maxcut(graph: nx.Graph) -> Dict[str, Any]:
    """
    Finds the exact maximum cut for a given graph using brute-force search.
    This method is computationally intensive and only suitable for small graphs (N < ~20).

    Args:
        graph (nx.Graph): The input NetworkX graph. Nodes must be integers from 0 to N-1.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        'max_cut_value': The maximum cut value found.
                        'max_cut_partitions': A list of partitions that achieve the maximum cut.
                        'num_nodes': Number of nodes in the graph.
    """
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return {'max_cut_value': 0, 'max_cut_partitions': [], 'num_nodes': 0}
    if num_nodes > 18: # Adjusted heuristic threshold for brute-force feasibility to give ILP some room
        print(f"Warning: Brute-force Max-Cut for {num_nodes} nodes is computationally expensive. "
              "Consider using an ILP solver for larger graphs. Skipping brute force.")
        return {
            'max_cut_value': -1, # Indicates not calculated by brute force
            'max_cut_partitions': [],
            'num_nodes': num_nodes,
            'solver_status': 'skipped_brute_force_too_slow'
        }

    max_cut_value = -1
    max_cut_partitions = []

    # Iterate through all possible partitions.
    # We fix the first node to be in partition 0 to avoid symmetric duplicates (2^N / 2 = 2^(N-1) partitions).
    # `bin(i)[2:].zfill(num_nodes-1)` generates binary strings for the remaining N-1 nodes.
    for i in range(2**(num_nodes - 1)):
        # Construct the partition for N nodes
        # The first node is always in set 0
        current_partition_str = bin(i)[2:].zfill(num_nodes - 1)
        current_partition = [0] + [int(bit) for bit in current_partition_str]
        
        current_value = calculate_cut_value(graph, current_partition)

        if current_value > max_cut_value:
            max_cut_value = current_value
            max_cut_partitions = [current_partition]
        elif current_value == max_cut_value:
            max_cut_partitions.append(current_partition)
            
    return {
        'max_cut_value': max_cut_value,
        'max_cut_partitions': max_cut_partitions,
        'num_nodes': num_nodes,
        'solver_status': 'brute_force_completed'
    }

def find_exact_maxcut_ilp(graph: nx.Graph) -> Dict[str, Any]:
    """
    Finds the exact maximum cut for a given graph using Integer Linear Programming (ILP).
    This method is more scalable than brute-force for larger graphs.

    Args:
        graph (nx.Graph): The input NetworkX graph. Nodes must be integers from 0 to N-1.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        'max_cut_value': The maximum cut value found.
                        'max_cut_partitions': A list containing one partition that achieves the maximum cut.
                                              (ILP typically returns only one optimal solution).
                        'num_nodes': Number of nodes in the graph.
                        'solver_status': Status of the ILP solver.
    """
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return {'max_cut_value': 0, 'max_cut_partitions': [], 'num_nodes': 0, 'solver_status': 'no_nodes'}

    # Create the ILP problem
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)

    # Decision variables: x_i = 0 if node i is in set S0, 1 if in set S1
    x = pulp.LpVariable.dicts("x", graph.nodes(), 0, 1, pulp.LpBinary)

    # Decision variables: y_ij = 1 if edge (i,j) is in the cut, 0 otherwise
    # y_ij = |x_i - x_j|  (This needs linearization for ILP)
    y = pulp.LpVariable.dicts("y", graph.edges(), 0, 1, pulp.LpBinary)

    # Objective function: Maximize the sum of y_ij for all edges
    prob += pulp.lpSum(y[edge] for edge in graph.edges()), "Total Cut Value"

    # Constraints for y_ij: y_ij = 1 if x_i != x_j
    # y_ij >= x_i - x_j
    # y_ij >= x_j - x_i
    # y_ij <= x_i + x_j
    # y_ij <= 2 - x_i - x_j
    for i, j in graph.edges():
        # Constraint 1: y_ij >= x_i - x_j
        prob += y[(i, j)] >= x[i] - x[j], f"y_ij_geq_xi_minus_xj_{i}_{j}"
        # Constraint 2: y_ij >= x_j - x_i
        prob += y[(i, j)] >= x[j] - x[i], f"y_ij_geq_xj_minus_xi_{i}_{j}"
        # Constraint 3: y_ij <= x_i + x_j (if x_i=0, x_j=0, then y_ij=0)
        prob += y[(i, j)] <= x[i] + x[j], f"y_ij_leq_xi_plus_xj_{i}_{j}"
        # Constraint 4: y_ij <= 2 - x_i - x_j (if x_i=1, x_j=1, then y_ij=0)
        prob += y[(i, j)] <= 2 - x[i] - x[j], f"y_ij_leq_2_minus_xi_minus_xj_{i}_{j}"

    # Solve the problem
    # Using the default solver (usually CBC, which comes with PuLP)
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output
    except Exception as e:
        print(f"Error during ILP solving: {e}")
        return {
            'max_cut_value': -1,
            'max_cut_partitions': [],
            'num_nodes': num_nodes,
            'solver_status': 'error_during_solve'
        }

    # Check solver status
    if prob.status != pulp.LpStatusOptimal:
        print(f"Warning: ILP solver did not find an optimal solution. Status: {pulp.LpStatus[prob.status]}")
        return {
            'max_cut_value': -1,
            'max_cut_partitions': [],
            'num_nodes': num_nodes,
            'solver_status': pulp.LpStatus[prob.status]
        }

    # Extract the solution
    max_cut_value = pulp.value(prob.objective)
    max_cut_partition = [int(pulp.value(x[node])) if pulp.value(x[node]) is not None else 0 for node in sorted(graph.nodes())] # Ensure order by node index

    return {
        'max_cut_value': int(max_cut_value),
        'max_cut_partitions': [max_cut_partition], # Return as a list, consistent with brute-force
        'num_nodes': num_nodes,
        'solver_status': pulp.LpStatus[prob.status]
    }

if __name__ == "__main__":
    # Example usage for brute-force:
    # Small graph (e.g., triangle)
    graph1 = nx.Graph()
    graph1.add_edges_from([(0, 1), (1, 2), (2, 0)])
    print("Graph 1 (Triangle) - Brute Force:")
    result1 = find_exact_maxcut(graph1)
    print(f"  Max Cut Value: {result1['max_cut_value']}")
    print(f"  Max Cut Partitions: {result1['max_cut_partitions']}")
    print(f"  Solver Status: {result1['solver_status']}")

    # Example usage for ILP:
    # A slightly larger graph, e.g., a cycle graph of 8 nodes
    graph_ilp_1 = nx.cycle_graph(8)
    print("\nGraph (Cycle N=8) - ILP:")
    result_ilp_1 = find_exact_maxcut_ilp(graph_ilp_1)
    print(f"  Max Cut Value: {result_ilp_1['max_cut_value']}")
    print(f"  Max Cut Partitions: {result_ilp_1['max_cut_partitions']}")
    print(f"  Solver Status: {result_ilp_1['solver_status']}")

    # Complete graph (K10) - Brute force would be too slow
    graph_ilp_2 = nx.complete_graph(10)
    print("\nGraph (Complete K10) - ILP:")
    result_ilp_2 = find_exact_maxcut_ilp(graph_ilp_2)
    print(f"  Max Cut Value: {result_ilp_2['max_cut_value']}")
    print(f"  Max Cut Partitions: {result_ilp_2['max_cut_partitions']}")
    print(f"  Solver Status: {result_ilp_2['solver_status']}")

    # Test with N=18 for brute-force limit
    graph_bf_limit = nx.cycle_graph(18)
    print("\nGraph (Cycle N=18) - Brute Force:")
    result_bf_limit = find_exact_maxcut(graph_bf_limit)
    print(f"  Max Cut Value: {result_bf_limit['max_cut_value']}")
    print(f"  Max Cut Partitions: {result_bf_limit['max_cut_partitions']}")
    print(f"  Solver Status: {result_bf_limit['solver_status']}")

    # Test with N=19 for brute-force limit (should be skipped)
    graph_bf_skipped = nx.cycle_graph(19)
    print("\nGraph (Cycle N=19) - Brute Force:")
    result_bf_skipped = find_exact_maxcut(graph_bf_skipped)
    print(f"  Max Cut Value: {result_bf_skipped['max_cut_value']}")
    print(f"  Max Cut Partitions: {result_bf_skipped['max_cut_partitions']}")
    print(f"  Solver Status: {result_bf_skipped['solver_status']}")

