import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.prompt import Prompt, IntPrompt

# Importazioni dai moduli locali
from common import graphs
from common.plotting import plot_qaoa_dashboard
from max_k_cut.components import build_max_k_cut_hamiltonians
from max_k_cut.circuit import create_max_k_cut_circuit, create_max_k_cut_sampling_circuit

def decode_bitstring(bitstring, n, k):
    """Converte una stringa di bit in un dizionario di colori."""
    colors = {}
    for i in range(n):
        node_bits = bitstring[i*k : (i+1)*k]
        try:
            # Trova l'indice del bit '1' (one-hot encoding)
            color = np.where(np.array(list(node_bits)) == '1')[0][0]
            colors[i] = color
        except IndexError:
            # Se nessun bit è 1 o ci sono più bit a 1, il nodo non ha un colore valido
            colors[i] = -1 
    return colors

def main():
    """Script principale interattivo per QAOA Max-k-Cut."""
    console = Console()
    console.rule("[bold magenta]QAOA Max-k-Cut: Selezione Interattiva[/bold magenta]")

    # ==========================================
    # 1. Configurazione k e Grafo
    # ==========================================
    k = IntPrompt.ask("Scegli il numero di colori (k)", default=3)
    
    console.print("\n[bold yellow]Scegli il tipo di grafo:[/bold yellow]")
    console.print("1. Ciclo (Cycle Graph)")
    console.print("2. Completo (Complete Graph)")
    console.print("3. Casuale (Random Erdos-Renyi)")
    console.print("4. Stella (Star Graph)")
    
    scelta = Prompt.ask("Inserisci il numero", choices=["1", "2", "3", "4"], default="2")
    
    if scelta == "1":
        n = IntPrompt.ask("Numero di nodi", default=3)
        graph = graphs.create_cycle_graph(n)
    elif scelta == "2":
        n = IntPrompt.ask("Numero di nodi", default=3)
        graph = graphs.create_complete_graph(n)
    elif scelta == "3":
        n = IntPrompt.ask("Numero di nodi", default=4)
        graph = graphs.create_random_graph(n)
    else:
        n = IntPrompt.ask("Numero di nodi", default=4)
        graph = graphs.create_star_graph(n)

    n_nodes = len(graph.nodes)
    n_qubits = n_nodes * k
    
    if n_qubits > 18:
        console.print("[bold red]Attenzione: Il numero totale di qubit è elevato. La simulazione potrebbe essere lenta.[/bold red]")

    graph_info = f"Nodi: {n_nodes}\nColori (k): {k}\nQubit Totali (n*k): {n_qubits}"
    console.print(Panel(graph_info, title="[bold green]Configurazione Problema[/bold green]", expand=False))

    # ==========================================
    # 2. Hamiltoniane e Circuiti
    # ==========================================
    penalty_weight = 2.0
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k, penalty_weight=penalty_weight)
    circuit = create_max_k_cut_circuit(graph, k, cost_h, mixer_h)
    sampling_circuit = create_max_k_cut_sampling_circuit(graph, k, cost_h, mixer_h)

    # ==========================================
    # 3. Ottimizzazione
    # ==========================================
    p = IntPrompt.ask("Numero di layer QAOA (p)", default=2)
    np.random.seed(42)
    params = np.array([np.random.uniform(0.01, 0.1, p) for _ in range(2)], requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.05)
    steps = 100
    cost_history = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Ottimizzazione...", total=steps)
        for i in range(steps):
            params = opt.step(circuit, params)
            current_cost = circuit(params)
            cost_history.append(current_cost)
            progress.update(task, advance=1, description=f"[cyan]Costo: {current_cost:.4f}[/cyan]")

    # ==========================================
    # 4. Risultati e Plotting
    # ==========================================
    probs = sampling_circuit(params)
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_qubits}b')
    node_colors = decode_bitstring(best_bitstring, n_nodes, k)
    
    result_info = f"Stringa Ottimale: {best_bitstring}\n"
    for node, color in node_colors.items():
        result_info += f"Nodo {node} -> Colore {color}\n"
    
    console.print(Panel(result_info, title="[bold green]Risultati[/bold green]", expand=False))
    plot_qaoa_dashboard(graph, k, probs, best_bitstring, node_colors=node_colors, cost_history=cost_history)

    # ==========================================
    # 5. Ispezione Manuale delle Soluzioni
    # ==========================================
    while True:
        cont = Prompt.ask("\nVuoi visualizzare un'altra soluzione specifica? (inserisci bitstring o 'no')", default="no")
        if cont.lower() == 'no':
            break
        
        if len(cont) != n_qubits or not set(cont).issubset({'0', '1'}):
            console.print(f"[bold red]Errore: Inserisci una stringa di {n_qubits} bit validi (0/1).[/bold red]")
            continue
        
        custom_node_colors = decode_bitstring(cont, n_nodes, k)
        plot_qaoa_dashboard(graph, k, probs, cont, node_colors=custom_node_colors, 
                            title=f"Visualizzazione Soluzione Manuale: {cont}")


if __name__ == "__main__":
    main()
