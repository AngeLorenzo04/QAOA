import pennylane as qml
from pennylane import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

# Importazioni dai moduli locali
from max_cut.utils import create_cycle_graph
from max_cut.plotting import plot_dashboard
from max_cut.components import build_maxcut_hamiltonians
from max_cut.circuit import create_qaoa_circuit, create_sampling_circuit


def main() -> None:
    """
    Script principale per la dimostrazione di QAOA applicato a Max-Cut.
    
    Questo script coordina l'intero flusso di lavoro:
    1. Definizione del problema (Grafo)
    2. Formulazione quantistica (Hamiltoniane)
    3. Inizializzazione dei parametri
    4. Ottimizzazione classica (Addestramento del circuito quantistico)
    5. Campionamento dei risultati e visualizzazione
    """
    # Inizializza la console Rich per una UI da terminale stilizzata
    console = Console()
    console.rule("[bold blue]Dimostrazione QAOA Max-Cut[/bold blue]")

    # ==========================================
    # 1. Definizione del problema geometrico (Grafo)
    # ==========================================
    graph = create_cycle_graph()
    n_wires = len(graph.nodes)

    graph_info = f"Nodi: {list(graph.nodes())}\nArchi: {list(graph.edges())}\nQubit Totali: {n_wires}"
    console.print(Panel(graph_info, title="[bold green]Definizione del Grafo[/bold green]", expand=False))

    # ==========================================
    # 2. Costruzione degli Operatori Quantistici
    # ==========================================
    # Mappiamo il problema del grafo classico in Hamiltoniane quantistiche.
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)

    # ==========================================
    # 3. Inizializzazione del Circuito Quantistico
    # ==========================================
    # Crea il circuito che andremo a ottimizzare. Restituisce il valore di aspettativa dell'Hamiltoniana di Costo.
    circuit = create_qaoa_circuit(graph, cost_h, mixer_h)

    # ==========================================
    # 4. Impostazione dei Parametri QAOA Iniziali
    # ==========================================
    p = 2  # La profondità del circuito QAOA (numero di layer). p più alto = più espressivo, ma più difficile da addestrare.
    
    # Inizializza gli angoli gamma (costo) e beta (mixer) casualmente tra 0 e pi.
    # Usiamo il wrapper numpy di PennyLane per consentire la differenziazione automatica.
    np.random.seed(42)
    gammas = np.random.uniform(0, np.pi, p, requires_grad=True)
    betas = np.random.uniform(0, np.pi, p, requires_grad=True)
    params = np.array([gammas, betas], requires_grad=True)

    console.print(f"[cyan]Parametri inizializzati per p={p} layer.[/cyan]\n")

    # ==========================================
    # 5. Ciclo di Ottimizzazione Classica
    # ==========================================
    # Usiamo l'ottimizzatore Adagrad per aggiornare iterativamente i parametri e minimizzare il costo.
    opt = qml.AdagradOptimizer(stepsize=0.5)
    steps = 40
    cost_history = []

    # Visualizza una barra di progresso durante l'addestramento
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[magenta]Ottimizzazione del Circuito QAOA...", total=steps)
        
        for i in range(steps):
            # Esegue un passo di gradiente per aggiornare gamma e beta
            params = opt.step(circuit, params)
            
            # Valuta la nuova funzione di costo (valore di aspettativa)
            current_cost = circuit(params)
            cost_history.append(current_cost)
            
            # Aggiorna visivamente la barra di progresso nel terminale
            progress.update(task, advance=1, description=f"[magenta]Ottimizzazione...[/magenta] (Costo: {current_cost:.4f})")

    console.print("\n[bold green]✔ Ottimizzazione Completata![/bold green]")

    # ==========================================
    # 6. Campionamento del Circuito Ottimizzato
    # ==========================================
    # Una volta ottenuti i parametri ottimali, eseguiamo una versione diversa del circuito 
    # che misura le probabilità effettive degli stati di base.
    sampling_circuit = create_sampling_circuit(graph, cost_h, mixer_h)
    probs = sampling_circuit(params)
    
    # ==========================================
    # 7. Estrazione dei Risultati
    # ==========================================
    # La migliore soluzione è la stringa di bit corrispondente all'indice con la probabilità più alta
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_wires}b')

    result_info = f"Stringa di bit ottimale: [bold yellow]{best_bitstring}[/bold yellow]\nProbabilità: {probs[best_idx]:.4f}\nCosto Finale: {cost_history[-1]:.4f}"
    console.print(Panel(result_info, title="[bold green]Risultati[/bold green]", expand=False))

    # ==========================================
    # 8. Dashboard di Visualizzazione
    # ==========================================
    console.print("[dim]Avvio della dashboard di visualizzazione... Chiudi la finestra per uscire.[/dim]")
    plot_dashboard(graph, probs, best_bitstring, cost_history, n_wires)


if __name__ == "__main__":
    main()
