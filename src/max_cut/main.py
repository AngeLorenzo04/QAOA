import pennylane as qml
from pennylane import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.prompt import Prompt, IntPrompt

# Importazioni dai moduli locali
from common import graphs
from common.plotting import plot_qaoa_dashboard
from max_cut.components import build_maxcut_hamiltonians
from max_cut.circuit import create_maxcut_circuit, create_maxcut_sampling_circuit


def get_user_graph_input(console):
    """
    Gestisce l'input dell'utente per la selezione del grafo.
    Restituisce il grafo scelto e il numero di qubit (n_wires).
    """
    console.print("\n[bold yellow]Scegli il tipo di grafo da analizzare:[/bold yellow]")
    console.print("1. Ciclo (Cycle Graph)")
    console.print("2. Completo (Complete Graph)")
    console.print("3. Casuale (Random Erdos-Renyi)")
    console.print("4. Petersen (10 nodi, 15 archi)")
    
    scelta = Prompt.ask("Inserisci il numero della tua scelta", choices=["1", "2", "3", "4"], default="1")
    
    if scelta == "1":
        n = IntPrompt.ask("Inserisci il numero di nodi", default=4)
        graph = graphs.create_cycle_graph(n)
        tipo = f"Ciclo con {n} nodi"
    elif scelta == "2":
        n = IntPrompt.ask("Inserisci il numero di nodi", default=4)
        graph = graphs.create_complete_graph(n)
        tipo = f"Completo con {n} nodi"
    elif scelta == "3":
        n = IntPrompt.ask("Inserisci il numero di nodi", default=5)
        graph = graphs.create_random_graph(n)
        tipo = f"Casuale con {n} nodi"
    else:
        graph = graphs.create_petersen_graph()
        tipo = "Grafo di Petersen"

    n_wires = len(graph.nodes)
    graph_info = f"Tipo: {tipo}\nNodi: {list(graph.nodes())}\nArchi: {len(graph.edges())}\nQubit Totali: {n_wires}"
    console.print(Panel(graph_info, title="[bold green]Configurazione Grafo[/bold green]", expand=False))
    return graph, n_wires

def setup_qaoa_operators(graph):
    """
    Costruisce gli operatori Hamiltoniani e il circuito QAOA.
    Restituisce cost_h, mixer_h, circuit.
    """
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)
    circuit = create_maxcut_circuit(graph, cost_h, mixer_h)
    return cost_h, mixer_h, circuit

def optimize_qaoa_parameters(console, circuit, p_layers, steps):
    """
    Inizializza e ottimizza i parametri QAOA.
    Restituisce i parametri ottimizzati e la cronologia dei costi.
    """
    np.random.seed(42)
    params = np.array([np.random.uniform(0, np.pi, p_layers) for _ in range(2)], requires_grad=True)
    
    opt = qml.AdagradOptimizer(stepsize=0.5)
    cost_history = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[magenta]Ottimizzazione in corso...", total=steps)
        for i in range(steps):
            params = opt.step(circuit, params)
            current_cost = circuit(params)
            cost_history.append(current_cost)
            progress.update(task, advance=1, description=f"[magenta]Costo: {current_cost:.4f}[/magenta]")
    return params, cost_history

def display_qaoa_results(console, graph, sampling_circuit, params, n_wires, cost_history):
    """
    Calcola e visualizza i risultati QAOA, inclusi il bitstring ottimale e il dashboard.
    """
    probs = sampling_circuit(params)
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_wires}b')

    result_info = f"Stringa Ottimale: [bold yellow]{best_bitstring}[/bold yellow]\nProbabilità: {probs[best_idx]:.4f}"
    console.print(Panel(result_info, title="[bold green]Risultati[/bold green]", expand=False))

    plot_qaoa_dashboard(graph, 2, probs, best_bitstring, cost_history=cost_history)

def manual_solution_inspection_maxcut(console, graph, sampling_circuit, params, n_wires):
    """
    Permette all'utente di ispezionare manualmente le soluzioni inserendo un bitstring.
    """
    probs = sampling_circuit(params) # Recalculate probs to avoid passing it around if not strictly necessary
    while True:
        cont = Prompt.ask("\nVuoi visualizzare un'altra soluzione specifica? (inserisci bitstring o 'no')", default="no")
        if cont.lower() == 'no':
            break
        
        if len(cont) != n_wires or not set(cont).issubset({'0', '1'}):
            console.print(f"[bold red]Errore: Inserisci una stringa di {n_wires} bit validi (0/1).[/bold red]")
            continue
        
        plot_qaoa_dashboard(graph, 2, probs, cont, title=f"Visualizzazione Soluzione Manuale: {cont}")


def main() -> None:
    """
    Script principale interattivo per la dimostrazione di QAOA Max-Cut.
    """
    console = Console()
    console.rule("[bold blue]QAOA Max-Cut: Selezione Interattiva[/bold blue]")

    # 1. Selezione del Grafo
    graph, n_wires = get_user_graph_input(console)

    # 2. Costruzione degli Operatori
    cost_h, mixer_h, circuit = setup_qaoa_operators(graph)

    # 3. Inizializzazione Parametri
    p_layers = IntPrompt.ask("Scegli il numero di layer QAOA (p)", default=2)
    steps = 40
    params, cost_history = optimize_qaoa_parameters(console, circuit, p_layers, steps)

    # 4. Risultati e Visualizzazione
    sampling_circuit = create_maxcut_sampling_circuit(graph, cost_h, mixer_h)
    display_qaoa_results(console, graph, sampling_circuit, params, n_wires, cost_history)

    # 5. Ispezione Manuale delle Soluzioni
    manual_solution_inspection_maxcut(console, graph, sampling_circuit, params, n_wires)





if __name__ == "__main__":
    main()
