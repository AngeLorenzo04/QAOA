import os
import sys
import pickle
import argparse
import networkx as nx
import numpy as np

# Rich imports for a high-quality CLI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

# Dynamic imports with fallback for PYTHONPATH variations
try:
    from src.qaoa.qaoa_runner import QAOARunner
    from src.data.graph_dataset_generator import load_graphs
    from src.data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
    from src.common.plotting import plot_qaoa_dashboard
except ModuleNotFoundError:
    try:
        from qaoa_runner import QAOARunner
        from data.graph_dataset_generator import load_graphs
        from data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
        from common.plotting import plot_qaoa_dashboard
    except ModuleNotFoundError:
        # Fallback if executing directly from src/qaoa/
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from qaoa.qaoa_runner import QAOARunner
        from data.graph_dataset_generator import load_graphs
        from data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
        from common.plotting import plot_qaoa_dashboard

def main():
    console = Console()
    console.rule("[bold purple]QAOA Runner: Selezione ed Esecuzione[/bold purple]")

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Esegui QAOA su un grafo selezionato con parametri personalizzati.")
    parser.add_argument('--non-interactive', action='store_true', help="Disabilita la modalità interattiva e usa solo gli argomenti CLI.")
    parser.add_argument('--graph-file', '-f', type=str, help="Percorso diretto a un file di grafo .gpickle.")
    parser.add_argument('--vertices', '-n', type=int, help="Filtra i grafi presenti per numero di nodi (N).")
    parser.add_argument('--density', '-d', type=float, help="Filtra i grafi presenti per densità (D).")
    parser.add_argument('--id', '-i', type=int, help="Filtra i grafi presenti per ID.")
    parser.add_argument('--optimizer', '-opt', type=str, choices=['COBYLA', 'SLSQP', 'GD'], help="Ottimizzatore classico da usare.")
    parser.add_argument('--layers', '-p', type=int, help="Numero di layer QAOA (p).")
    parser.add_argument('--shots', '-s', type=int, default=1024, help="Numero di shots per la simulazione (default: 1024).")
    parser.add_argument('--plot', action='store_true', help="Mostra la dashboard grafica al termine.")
    args = parser.parse_args()

    # Determine default directory for graphs
    graphs_dir = "data/generated_graphs"
    graph = None
    graph_metadata = {}

    # 1. Load Graph
    if args.graph_file:
        # User specified a direct file
        if not os.path.exists(args.graph_file):
            console.print(f"[bold red]Errore: Il file del grafo '{args.graph_file}' non esiste.[/bold red]")
            sys.exit(1)
        try:
            with open(args.graph_file, 'rb') as f:
                graph = pickle.load(f)
            graph_metadata = {
                'n_vertices': graph.graph.get('n_vertices', len(graph.nodes)),
                'density_edges': graph.graph.get('density_edges', nx.density(graph)),
                'id': graph.graph.get('id', 'custom'),
                'seed': graph.graph.get('seed', 'custom'),
                'filepath': args.graph_file
            }
            console.print(f"[green]Caricato grafo personalizzato da {args.graph_file}[/green]")
        except Exception as e:
            console.print(f"[bold red]Errore nel caricamento del file del grafo: {e}[/bold red]")
            sys.exit(1)
    else:
        # Load from dataset
        graphs_list = load_graphs(graphs_dir)
        
        # Fallback: if no graphs are present, offer to generate some
        if not graphs_list:
            console.print("[yellow]Nessun grafo trovato nella directory di default 'data/generated_graphs'.[/yellow]")
            if args.non_interactive:
                console.print("[bold red]Errore: Impossibile continuare in modalità non-interattiva senza grafi.[/bold red]")
                sys.exit(1)
            
            gen_choice = Prompt.ask("Vuoi generare automaticamente alcuni grafi di prova?", choices=["si", "no"], default="si")
            if gen_choice.lower() in ["si", "s"]:
                console.print("[green]Generazione in corso...[/green]")
                from src.data.graph_dataset_generator import generate_and_save_graphs
                try:
                    generate_and_save_graphs([4, 8], [0.25, 0.50], 2, graphs_dir)
                    graphs_list = load_graphs(graphs_dir)
                except Exception as e:
                    console.print(f"[bold red]Errore durante la generazione dei grafi: {e}[/bold red]")
                    sys.exit(1)
            else:
                console.print("[bold red]Uscita programmata. Genera i grafi o passa un file con --graph-file.[/bold red]")
                sys.exit(0)

        # Selection logic (interactive or argument-driven filter)
        if args.non_interactive:
            # Match directly based on filters
            filtered = graphs_list
            if args.vertices is not None:
                filtered = [g for g in filtered if g['n_vertices'] == args.vertices]
            if args.density is not None:
                filtered = [g for g in filtered if abs(g['density_edges'] - args.density) < 1e-5]
            if args.id is not None:
                filtered = [g for g in filtered if g['id'] == args.id]
            
            if not filtered:
                console.print("[bold red]Errore: Nessun grafo soddisfa i filtri specificati in modalità non-interattiva.[/bold red]")
                sys.exit(1)
            
            selected_entry = filtered[0]
            graph = selected_entry['graph']
            graph_metadata = selected_entry
        else:
            # Interactive selection
            # Step 1: Vertices (N)
            available_n = sorted(list(set(g['n_vertices'] for g in graphs_list)))
            if args.vertices in available_n:
                n_selected = args.vertices
            else:
                n_choices = [str(n) for n in available_n]
                n_selected = int(Prompt.ask("Seleziona il numero di nodi (N)", choices=n_choices, default=n_choices[0]))
            
            # Filter by N
            graphs_filtered_n = [g for g in graphs_list if g['n_vertices'] == n_selected]
            
            # Step 2: Density (D)
            available_d = sorted(list(set(g['density_edges'] for g in graphs_filtered_n)))
            if args.density is not None and any(abs(d - args.density) < 1e-5 for d in available_d):
                d_selected = next(d for d in available_d if abs(d - args.density) < 1e-5)
            else:
                d_choices = [f"{d:.2f}" for d in available_d]
                d_selected_str = Prompt.ask("Seleziona la densità degli archi (D)", choices=d_choices, default=d_choices[0])
                d_selected = float(d_selected_str)
                
            # Filter by N and D
            graphs_filtered_nd = [g for g in graphs_filtered_n if abs(g['density_edges'] - d_selected) < 1e-5]
            
            # Step 3: Graph ID / Seed
            if args.id is not None and any(g['id'] == args.id for g in graphs_filtered_nd):
                selected_entry = next(g for g in graphs_filtered_nd if g['id'] == args.id)
            else:
                table = Table(title=f"Grafi disponibili (N={n_selected}, D={d_selected:.2f})")
                table.add_column("Indice", style="cyan", justify="right")
                table.add_column("ID Grafo", style="magenta")
                table.add_column("Seed", style="green")
                table.add_column("Nodi", style="yellow")
                table.add_column("Archi", style="blue")
                
                for idx, g in enumerate(graphs_filtered_nd):
                    table.add_row(
                        str(idx),
                        str(g['id']),
                        str(g['seed']),
                        str(g['graph'].number_of_nodes()),
                        str(g['graph'].number_of_edges())
                    )
                console.print(table)
                
                idx_choices = [str(i) for i in range(len(graphs_filtered_nd))]
                idx_selected = int(Prompt.ask("Seleziona l'indice del grafo", choices=idx_choices, default="0"))
                selected_entry = graphs_filtered_nd[idx_selected]
            
            graph = selected_entry['graph']
            graph_metadata = selected_entry

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # 2. Select Optimizer and layers
    if args.optimizer:
        optimizer = args.optimizer
    elif args.non_interactive:
        optimizer = 'COBYLA'
    else:
        optimizer = Prompt.ask("Scegli l'ottimizzatore classico", choices=['COBYLA', 'SLSQP', 'GD'], default='COBYLA')

    if args.layers:
        p_layers = args.layers
    elif args.non_interactive:
        p_layers = 1
    else:
        p_layers = IntPrompt.ask("Scegli il numero di layer QAOA (p)", default=1)

    # 3. Present configuration
    config_info = (
        f"Nodi (N): {n_nodes}\n"
        f"Archi: {n_edges}\n"
        f"Densità: {graph_metadata.get('density_edges', 0.0):.2f}\n"
        f"ID Grafo: {graph_metadata.get('id', 'N/A')}\n"
        f"Seed: {graph_metadata.get('seed', 'N/A')}\n"
        f"Layer QAOA (p): {p_layers}\n"
        f"Ottimizzatore: {optimizer}\n"
        f"Shots: {args.shots}"
    )
    console.print(Panel(config_info, title="[bold green]Configurazione QAOA[/bold green]", expand=False))

    # 4. Run exact MaxCut calculation for comparison if not already done
    exact_maxcut_val = graph.graph.get('exact_max_cut_value', -1)
    if exact_maxcut_val == -1:
        with console.status("[bold yellow]Calcolo del Max-Cut esatto classica...[/bold yellow]"):
            if n_nodes <= 16:
                exact_res = find_exact_maxcut(graph)
            else:
                exact_res = find_exact_maxcut_ilp(graph)
            exact_maxcut_val = exact_res['max_cut_value']
            graph.graph['exact_max_cut_value'] = exact_maxcut_val

    # 5. Run QAOA
    runner = QAOARunner(graph, p_value=p_layers)
    
    with console.status(f"[bold magenta]Esecuzione di QAOA con {optimizer}...[/bold magenta]"):
        try:
            qaoa_results = runner.run(
                optimizer_method=optimizer,
                shots=args.shots,
                max_optimization_iterations=100
            )
        except Exception as e:
            console.print(f"[bold red]Errore durante l'esecuzione di QAOA: {e}[/bold red]")
            sys.exit(1)

    # 6. Show results
    best_bitstring = qaoa_results['best_measured_bitstring']
    expected_cut = qaoa_results['qaoa_expected_cut_value']
    best_cut = qaoa_results['best_measured_cut_value']
    iterations = qaoa_results['metrics']['optimization_iterations']
    termination_reason = qaoa_results['metrics']['termination_reason']
    
    approx_ratio_expected = expected_cut / exact_maxcut_val if exact_maxcut_val > 0 else 0.0
    approx_ratio_best = best_cut / exact_maxcut_val if exact_maxcut_val > 0 else 0.0

    result_text = (
        f"Taglio Massimo Esatto (Classico): {exact_maxcut_val}\n"
        f"Taglio QAOA Atteso: {expected_cut:.4f} (Ratio: {approx_ratio_expected:.4f})\n"
        f"Miglior Taglio Misurato: {best_cut} (Ratio: {approx_ratio_best:.4f})\n"
        f"Migliore Stringa di Bit: {best_bitstring}\n"
        f"Iterazioni Ottimizzatore: {iterations}\n"
        f"Motivo Termine: {termination_reason}"
    )
    console.print(Panel(result_text, title="[bold green]Risultati Ottimizzazione[/bold green]", expand=False))

    # 7. Dashboard visualization
    show_plot = args.plot
    if not args.non_interactive and not args.plot:
        plot_choice = Prompt.ask("\nVuoi visualizzare la dashboard con i grafici dei risultati?", choices=["si", "no"], default="si")
        if plot_choice.lower() in ["si", "s"]:
            show_plot = True

    if show_plot:
        # Convert quasi-distribution to a probability array of size 2^n_nodes for plotting
        probs = np.zeros(2**n_nodes)
        for bitstr, val in qaoa_results['quasi_distribution'].items():
            state_idx = int(bitstr, 2)
            probs[state_idx] = val
        
        console.print("[cyan]Visualizzazione dashboard in corso...[/cyan]")
        try:
            plot_qaoa_dashboard(
                graph,
                k=2,
                probs=probs,
                best_bitstring=best_bitstring,
                cost_history=qaoa_results['metrics']['optimization_history']['fun'],
                title=f"QAOA Max-Cut - N={n_nodes}, p={p_layers}, opt={optimizer} (Best Cut: {best_cut}/{exact_maxcut_val})"
            )
        except Exception as e:
            console.print(f"[bold red]Errore nella visualizzazione grafica: {e}[/bold red]")

if __name__ == "__main__":
    main()
