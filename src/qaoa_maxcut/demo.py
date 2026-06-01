from qaoa_maxcut.graph_definition import create_cycle_graph


def main() -> None:
    graph = create_cycle_graph()
    wires = range(len(graph.nodes))

    print("Wires:", list(wires))
    print("Nodi del grafo:", list(graph.nodes()))
    print("Archi del grafo:", list(graph.edges()))


if __name__ == "__main__":
    main()