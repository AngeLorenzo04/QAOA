from qaoa_maxcut.graph_definition import create_cycle_graph
from qaoa_maxcut.plotting import plot_graph
from qaoa_maxcut.qaoa_components import build_maxcut_hamiltonians
from qaoa_maxcut.circuit import create_qaoa_circuit


def main() -> None:
    graph = create_cycle_graph()
    wires = range(len(graph.nodes))

    print("Wires:", list(wires))
    print("Nodi del grafo:", list(graph.nodes()))
    print("Archi del grafo:", list(graph.edges()))

    plot_graph(graph)

    cost_h, mixer_h = build_maxcut_hamiltonians(graph)

    print("\nCost Hamiltonian:")
    print(cost_h)

    print("\nMixer Hamiltonian:")
    print(mixer_h)

    circuit = create_qaoa_circuit(graph, cost_h, mixer_h)

    gamma = 0.5
    beta = 0.3

    energy = circuit(gamma, beta)

    print(f"\nExpectation value of cost Hamiltonian for gamma={gamma}, beta={beta}:")
    print(energy)


if __name__ == "__main__":
    main()