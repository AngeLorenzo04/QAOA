import pennylane as qml
from pennylane import numpy as np
from qaoa_maxcut.graph_definition import create_cycle_graph
from qaoa_maxcut.plotting import plot_graph, plot_probabilities, plot_result_graph
from qaoa_maxcut.qaoa_components import build_maxcut_hamiltonians
from qaoa_maxcut.circuit import create_qaoa_circuit, create_sampling_circuit


def main() -> None:
    # 1. Define the graph for the Max-Cut problem (a 4-node cycle)
    graph = create_cycle_graph()
    n_wires = len(graph.nodes)
    wires = range(n_wires)

    print("Wires:", list(wires))
    print("Nodi del grafo:", list(graph.nodes()))
    print("Archi del grafo:", list(graph.edges()))

    # Visualize the problem graph
    plot_graph(graph)

    # 2. Build the Cost and Mixer Hamiltonians for Max-Cut
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)

    print("\nCost Hamiltonian:")
    print(cost_h)

    print("\nMixer Hamiltonian:")
    print(mixer_h)

    # 3. Create the QAOA circuit (Expectation value for optimization)
    circuit = create_qaoa_circuit(graph, cost_h, mixer_h)

    # 4. Set QAOA parameters
    p = 2  # Number of QAOA layers
    
    # Initialize gamma and beta parameters randomly
    np.random.seed(42)
    gammas = np.random.uniform(0, np.pi, p, requires_grad=True)
    betas = np.random.uniform(0, np.pi, p, requires_grad=True)
    params = np.array([gammas, betas], requires_grad=True)

    # 5. Classical Optimization Loop
    opt = qml.AdagradOptimizer(stepsize=0.5)
    steps = 40

    print(f"\nStarting optimization for p={p} layers...")
    
    for i in range(steps):
        params = opt.step(circuit, params)
        
        if (i + 1) % 5 == 0:
            current_cost = circuit(params)
            print(f"Step {i+1:3d}: Cost = {current_cost:.6f}")

    print("\nOptimization complete.")

    # 6. Sampling and Result Visualization
    sampling_circuit = create_sampling_circuit(graph, cost_h, mixer_h)
    probs = sampling_circuit(params)

    # Display the probability distribution
    plot_probabilities(probs, n_wires)

    # 7. Find and visualize the best cut
    # The best cut is the bitstring with the highest probability
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_wires}b')

    print(f"\nBest bitstring found: {best_bitstring}")
    print(f"Probability: {probs[best_idx]:.4f}")

    # Visualize the graph partition
    plot_result_graph(graph, best_bitstring)


if __name__ == "__main__":
    main()
