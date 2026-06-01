import pennylane as qml
from pennylane import numpy as np
from qaoa_maxcut.graph_definition import create_cycle_graph
from qaoa_maxcut.plotting import plot_graph, plot_probabilities, plot_result_graph, plot_cost_history
from qaoa_maxcut.qaoa_components import build_maxcut_hamiltonians
from qaoa_maxcut.circuit import create_qaoa_circuit, create_sampling_circuit


def main() -> None:
    # 1. Define the graph
    graph = create_cycle_graph()
    n_wires = len(graph.nodes)

    print("Nodi del grafo:", list(graph.nodes()))
    print("Archi del grafo:", list(graph.edges()))

    # 2. Build Hamiltonians
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)

    # 3. Create circuits
    circuit = create_qaoa_circuit(graph, cost_h, mixer_h)

    # 4. Set parameters
    p = 2
    np.random.seed(42)
    gammas = np.random.uniform(0, np.pi, p, requires_grad=True)
    betas = np.random.uniform(0, np.pi, p, requires_grad=True)
    params = np.array([gammas, betas], requires_grad=True)

    # 5. Optimization Loop with History Tracking
    opt = qml.AdagradOptimizer(stepsize=0.5)
    steps = 40
    cost_history = []

    print(f"\nStarting optimization for p={p} layers...")
    
    for i in range(steps):
        params = opt.step(circuit, params)
        current_cost = circuit(params)
        cost_history.append(current_cost)
        
        if (i + 1) % 5 == 0:
            print(f"Step {i+1:3d}: Cost = {current_cost:.6f}")

    # Visualize optimization convergence
    plot_cost_history(cost_history)

    # 6. Sampling
    sampling_circuit = create_sampling_circuit(graph, cost_h, mixer_h)
    probs = sampling_circuit(params)
    plot_probabilities(probs, n_wires)

    # 7. Decision Boundary Visualization
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_wires}b')

    print(f"\nBest bitstring found: {best_bitstring}")
    plot_result_graph(graph, best_bitstring)


if __name__ == "__main__":
    main()
