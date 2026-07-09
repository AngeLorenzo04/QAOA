import json
import os
import sys

# Aggiunge la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from src.visualization.plotter import plot_approximation_ratio_vs_params, plot_optimizer_convergence, plot_probability_distribution, plot_graph_with_cut

try:
    from rich.console import Console
    from rich.prompt import Prompt
    console = Console()
except ImportError:
    print("Per usare la modalità interattiva, installa 'rich': pip install rich")
    sys.exit(1)

SUMMARY_FILE = "data/benchmarking_results/qaoa_benchmarking_summary.json"

def ask_choice(prompt_text: str, choices: list, default: str) -> str:
    return Prompt.ask(prompt_text, choices=choices, default=default)

def handle_specific_graph_menu(data):
    while True:
        filtered = data
        console.rule("[bold magenta]Filtra i Risultati del Benchmark (oppure premi 'q' per tornare indietro)[/bold magenta]")

        # Filtro N
        available_n = sorted(list(set(m['graph_metadata']['n_vertices'] for m in filtered)))
        if len(available_n) > 1:
            choices_n = [str(n) for n in available_n] + ["q"]
            ans_n = ask_choice("Scegli il numero di nodi (N)", choices_n, choices_n[0])
            if ans_n == 'q': return
            filtered = [m for m in filtered if m['graph_metadata']['n_vertices'] == int(ans_n)]

        # Filtro D
        available_d = sorted(list(set(m['graph_metadata']['density_edges'] for m in filtered)))
        if len(available_d) > 1:
            choices_d = [f"{d:.2f}" for d in available_d] + ["q"]
            ans_d = ask_choice("Scegli la densità (D)", choices_d, choices_d[0])
            if ans_d == 'q': return
            filtered = [m for m in filtered if abs(m['graph_metadata']['density_edges'] - float(ans_d)) < 1e-6]

        # Filtro p
        available_p = sorted(list(set(m['qaoa_config']['p_value'] for m in filtered)))
        if len(available_p) > 1:
            choices_p = [str(p) for p in available_p] + ["q"]
            ans_p = ask_choice("Scegli il layer (p)", choices_p, choices_p[0])
            if ans_p == 'q': return
            filtered = [m for m in filtered if m['qaoa_config']['p_value'] == int(ans_p)]

        # Filtro Optimizer
        available_opt = sorted(list(set(m['qaoa_config'].get('optimizer', 'COBYLA') for m in filtered)))
        if len(available_opt) > 1:
            choices_opt = available_opt + ["q"]
            ans_opt = ask_choice("Scegli l'algoritmo di ottimizzazione", choices_opt, choices_opt[0])
            if ans_opt == 'q': return
            filtered = [m for m in filtered if m['qaoa_config'].get('optimizer', 'COBYLA') == ans_opt]

        # Filtro ID
        available_id = sorted(list(set(m['graph_metadata']['id'] for m in filtered)))
        if len(available_id) > 1:
            choices_id = [str(i) for i in available_id] + ["q"]
            ans_id = ask_choice("Scegli l'ID del grafo", choices_id, choices_id[0])
            if ans_id == 'q': return
            filtered = [m for m in filtered if m['graph_metadata']['id'] == int(ans_id)]

        if not filtered:
            console.print("[bold red]Nessun risultato trovato con i parametri selezionati.[/bold red]")
            continue

        sample_result = filtered[0]
        n = sample_result['graph_metadata']['n_vertices']
        d = sample_result['graph_metadata']['density_edges']
        p = sample_result['qaoa_config']['p_value']
        opt = sample_result['qaoa_config'].get('optimizer', 'COBYLA')
        gid = sample_result['graph_metadata']['id']

        while True:
            console.rule(f"[bold cyan]Risultato Selezionato: N={n}, D={d:.2f}, p={p}, opt={opt}, ID={gid}[/bold cyan]")
            
            console.print("\n1. [bold yellow]Convergenza dell'Ottimizzatore[/bold yellow] (Traccia del costo)")
            console.print("2. [bold yellow]Distribuzione di Probabilità[/bold yellow] (Stati misurati)")
            console.print("3. [bold yellow]Visualizza il Grafo (NetworkX)[/bold yellow] (Rete e Max Cut trovato da QAOA)")
            console.print("4. [bold red]Scegli un altro grafo (Torna Indietro)[/bold red]")
            console.print("5. [bold red]Torna al Menu Principale[/bold red]")
            
            plot_choice = ask_choice("\nQuale grafico vuoi generare?", choices=["1", "2", "3", "4", "5"], default="1")

            if plot_choice == "4":
                break # Esce da questo while, tornando alla selezione del grafo
            elif plot_choice == "5":
                return # Esce dalla funzione handle_specific_graph_menu, tornando al main()

            if plot_choice == "1":
                console.print(f"\n[cyan]Generazione grafico a schermo: Convergenza Ottimizzatore...[/cyan]")
                plot_optimizer_convergence(
                    sample_result['metrics']['optimization_history'], 
                    f"Convergenza Ottimizzatore (N={n}, D={d:.2f}, p={p}, {opt})",
                    filepath=None 
                )
            elif plot_choice == "2":
                if 'quasi_distribution' in sample_result['qaoa_results']:
                    console.print(f"\n[cyan]Generazione grafico a schermo: Distribuzione di Probabilità...[/cyan]")
                    plot_probability_distribution(
                        sample_result['qaoa_results']['quasi_distribution'],
                        title=f"Distribuzione di Probabilità (N={n}, D={d:.2f}, p={p}, {opt})",
                        optimal_bitstring=sample_result['qaoa_results'].get('best_measured_bitstring'),
                        filepath=None 
                    )
                else:
                    console.print("\n[bold red]Attenzione:[/bold red] I dati 'quasi_distribution' non sono stati salvati nel JSON.")
                    console.print("Esegui nuovamente il benchmark per visualizzarlo.")
            elif plot_choice == "3":
                seed = sample_result['graph_metadata']['seed']
                graph_filepath = f"data/generated_graphs/graph_n{n}_d{d:.2f}_id{gid}_seed{seed}.gpickle"
                
                if not os.path.exists(graph_filepath):
                    console.print(f"\n[bold red]Errore:[/bold red] Impossibile trovare il file del grafo in {graph_filepath}")
                else:
                    try:
                        with open(graph_filepath, 'rb') as pf:
                            graph = pickle.load(pf)
                            
                        best_bitstring = sample_result['qaoa_results'].get('best_measured_bitstring')
                        partition = None
                        if best_bitstring:
                            partition = [int(bit) for bit in best_bitstring]
                            
                        console.print(f"\n[cyan]Apertura grafo a schermo (Taglio trovato da QAOA: {best_bitstring})...[/cyan]")
                        plot_graph_with_cut(
                            graph, 
                            partition=partition, 
                            title=f"Grafo N={n}, D={d:.2f}, ID={gid} - Taglio Migliore QAOA: {best_bitstring}",
                            filepath=None
                        )
                    except Exception as e:
                        console.print(f"\n[bold red]Errore durante il caricamento/visualizzazione del grafo:[/bold red] {e}")


def main():
    if not os.path.exists(SUMMARY_FILE):
        console.print(f"[bold red]File {SUMMARY_FILE} non trovato. Assicurati che il benchmarking abbia finito di salvare i risultati.[/bold red]")
        return

    with open(SUMMARY_FILE, 'r') as f:
        data = json.load(f)
        
    if not data:
        console.print("[bold red]Il file dei risultati è vuoto.[/bold red]")
        return

    while True:
        console.rule("[bold blue]Menu Principale: Visualizzazione Risultati di Benchmarking[/bold blue]")
        console.print("1. [bold green]Grafici Globali Aggregati[/bold green] (Approximation Ratio vs N, p, Densità)")
        console.print("2. [bold magenta]Grafici Specifici per un Singolo Benchmark[/bold magenta] (Convergenza, Distribuzione)")
        console.print("3. [bold red]Esci[/bold red]")

        scelta_principale = ask_choice("\nCosa vuoi visualizzare?", choices=["1", "2", "3"], default="1")

        if scelta_principale == "3":
            console.print("[bold cyan]Uscita in corso...[/bold cyan]")
            break

        if scelta_principale == "1":
            console.print("\nGenerazione grafico: Approximation Ratio vs Numero di Vertici (N)...")
            plot_approximation_ratio_vs_params(
                data, 
                'n_vertices', 
                "Approximation Ratio vs Numero di Nodi",
                filepath="data/benchmarking_results/plot_approx_vs_N.png"
            )
            console.print("Generazione grafico: Approximation Ratio vs Profondità del circuito (p)...")
            plot_approximation_ratio_vs_params(
                data, 
                'p_value', 
                "Approximation Ratio vs Layer p",
                filepath="data/benchmarking_results/plot_approx_vs_p.png"
            )
            console.print("Generazione grafico: Approximation Ratio vs Densità...")
            plot_approximation_ratio_vs_params(
                data, 
                'density_edges', 
                "Approximation Ratio vs Densità Archi",
                filepath="data/benchmarking_results/plot_approx_vs_density.png"
            )
            console.print("\n[bold green]Tutte le dashboard globali sono state generate e salvate in 'data/benchmarking_results/'![/bold green]\n")
        
        elif scelta_principale == "2":
            handle_specific_graph_menu(data)


if __name__ == "__main__":
    main()
