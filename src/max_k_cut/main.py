import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

# Importazioni dai moduli locali
from max_k_cut.components import build_max_k_cut_hamiltonians
from max_k_cut.circuit import create_max_k_cut_circuit, create_max_k_cut_sampling_circuit
from max_k_cut.utils import create_cycle_graph, create_complete_graph
from max_k_cut.plotting import plot_k_cut_dashboard

def decode_bitstring(bitstring, n, k):
    """
    Converte una stringa di bit (lunghezza n*k) in un dizionario di colori dei nodi.
    """
    colors = {}
    for i in range(n):
        node_bits = bitstring[i*k : (i+1)*k]
        try:
            # Trova l'indice del bit '1' nella sezione del nodo i
            color = np.where(np.array(list(node_bits)) == '1')[0][0]
            colors[i] = color
        except IndexError:
            colors[i] = -1 
    return colors

def main():
    """
    Script principale per la dimostrazione di QAOA applicato a Max-k-Cut.
    """
    console = Console()
    console.rule("[bold magenta]Dimostrazione QAOA Max-k-Cut[/bold magenta]")

    # ==========================================
    # 1. Configurazione del Problema
    # ==========================================
    k = 3 # Numero di partizioni (colori)
    n_nodes = 3 # Numero di nodi nel grafo
    
    # Creiamo un grafo completo (triangolo) per k=3
    graph = create_complete_graph(n_nodes)
    n_qubits = n_nodes * k

    graph_info = (f"Tipo: Grafo Completo (Clique)\n"
                  f"Nodi: {n_nodes}\n"
                  f"Colori (k): {k}\n"
                  f"Qubit Totali (n*k): {n_qubits}")
    console.print(Panel(graph_info, title="[bold green]Configurazione Problema[/bold green]", expand=False))

    # ==========================================
    # 2. Costruzione delle Hamiltoniane
    # ==========================================
    # Peso di penalità per forzare 'one-hot' encoding per ogni nodo
    penalty_weight = 2.0
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k, penalty_weight=penalty_weight)

    # ==========================================
    # 3. Creazione dei Circuiti
    # ==========================================
    circuit = create_max_k_cut_circuit(graph, k, cost_h, mixer_h)
    sampling_circuit = create_max_k_cut_sampling_circuit(graph, k, cost_h, mixer_h)

    # ==========================================
    # 4. Ottimizzazione Classica
    # ==========================================
    n_layers = 4
    np.random.seed(42)
    # Inizializziamo i parametri gamma e beta
    params = np.array([np.random.uniform(0.01, 0.1, n_layers) for _ in range(2)], requires_grad=True)
    
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
        
        task = progress.add_task("[cyan]Ottimizzazione QAOA...", total=steps)
        
        for i in range(steps):
            params = opt.step(circuit, params)
            current_cost = circuit(params)
            cost_history.append(current_cost)
            progress.update(task, advance=1, description=f"[cyan]Ottimizzando...[/cyan] (Costo: {current_cost:.4f})")

    console.print("\n[bold green]✔ Ottimizzazione Completata![/bold green]")

    # ==========================================
    # 5. Campionamento e Risultati
    # ==========================================
    probs = sampling_circuit(params)
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_qubits}b')
    
    node_colors = decode_bitstring(best_bitstring, n_nodes, k)
    
    result_info = f"Stringa di bit ottimale: [bold yellow]{best_bitstring}[/bold yellow]\n"
    result_info += "Assegnazione colori:\n"
    for node, color in node_colors.items():
        result_info += f"  Nodo {node} -> Colore {color if color != -1 else 'INVALIDO'}\n"
    
    console.print(Panel(result_info, title="[bold green]Risultati Finali[/bold green]", expand=False))

    # ==========================================
    # 6. Dashboard di Visualizzazione
    # ==========================================
    console.print("[dim]Avvio della dashboard k-Cut... Chiudi la finestra per uscire.[/dim]")
    plot_k_cut_dashboard(graph, k, probs, best_bitstring, node_colors, cost_history)

if __name__ == "__main__":
    main()
