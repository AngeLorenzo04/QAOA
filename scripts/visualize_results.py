import json
import os
import sys

# Aggiunge la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.plotter import plot_approximation_ratio_vs_params, plot_optimizer_convergence, plot_probability_distribution

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

def main():
    if not os.path.exists(SUMMARY_FILE):
        console.print(f"[bold red]File {SUMMARY_FILE} non trovato. Assicurati che il benchmarking abbia finito di salvare i risultati.[/bold red]")
        return

    with open(SUMMARY_FILE, 'r') as f:
        data = json.load(f)
        
    if not data:
        console.print("[bold red]Il file dei risultati è vuoto.[/bold red]")
        return

    console.rule("[bold blue]Menu Visualizzazione Risultati di Benchmarking[/bold blue]")
    console.print("1. [bold green]Grafici Globali Aggregati[/bold green] (Approximation Ratio vs N, p, Densità)")
    console.print("2. [bold magenta]Grafici Specifici per un Singolo Benchmark[/bold magenta] (Convergenza, Distribuzione)")

    scelta_principale = ask_choice("\nCosa vuoi visualizzare?", choices=["1", "2"], default="1")

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
        console.print("\n[bold green]Tutte le dashboard globali sono state generate e salvate in 'data/benchmarking_results/'![/bold green]")
        return

    # === Modalità Grafico Specifico ===
    filtered = data
    console.rule("[bold magenta]Filtra i Risultati del Benchmark[/bold magenta]")

    # Filtro N
    available_n = sorted(list(set(m['graph_metadata']['n_vertices'] for m in filtered)))
    if len(available_n) > 1:
        choices_n = [str(n) for n in available_n]
        ans_n = ask_choice("Scegli il numero di nodi (N)", choices_n, choices_n[0])
        filtered = [m for m in filtered if m['graph_metadata']['n_vertices'] == int(ans_n)]

    # Filtro D
    available_d = sorted(list(set(m['graph_metadata']['density_edges'] for m in filtered)))
    if len(available_d) > 1:
        choices_d = [f"{d:.2f}" for d in available_d]
        ans_d = ask_choice("Scegli la densità (D)", choices_d, choices_d[0])
        filtered = [m for m in filtered if abs(m['graph_metadata']['density_edges'] - float(ans_d)) < 1e-6]

    # Filtro p
    available_p = sorted(list(set(m['qaoa_config']['p_value'] for m in filtered)))
    if len(available_p) > 1:
        choices_p = [str(p) for p in available_p]
        ans_p = ask_choice("Scegli il layer (p)", choices_p, choices_p[0])
        filtered = [m for m in filtered if m['qaoa_config']['p_value'] == int(ans_p)]

    # Filtro Optimizer
    available_opt = sorted(list(set(m['qaoa_config'].get('optimizer', 'COBYLA') for m in filtered)))
    if len(available_opt) > 1:
        choices_opt = available_opt
        ans_opt = ask_choice("Scegli l'algoritmo di ottimizzazione", choices_opt, choices_opt[0])
        filtered = [m for m in filtered if m['qaoa_config'].get('optimizer', 'COBYLA') == ans_opt]

    # Filtro ID
    available_id = sorted(list(set(m['graph_metadata']['id'] for m in filtered)))
    if len(available_id) > 1:
        choices_id = [str(i) for i in available_id]
        ans_id = ask_choice("Scegli l'ID del grafo", choices_id, choices_id[0])
        filtered = [m for m in filtered if m['graph_metadata']['id'] == int(ans_id)]

    if not filtered:
        console.print("[bold red]Nessun risultato trovato con i parametri selezionati.[/bold red]")
        return

    sample_result = filtered[0]
    n = sample_result['graph_metadata']['n_vertices']
    d = sample_result['graph_metadata']['density_edges']
    p = sample_result['qaoa_config']['p_value']
    opt = sample_result['qaoa_config'].get('optimizer', 'COBYLA')
    gid = sample_result['graph_metadata']['id']

    console.rule(f"[bold cyan]Risultato Selezionato: N={n}, D={d:.2f}, p={p}, opt={opt}, ID={gid}[/bold cyan]")
    
    console.print("\n1. [bold yellow]Convergenza dell'Ottimizzatore[/bold yellow] (Traccia del costo)")
    console.print("2. [bold yellow]Distribuzione di Probabilità[/bold yellow] (Stati misurati)")
    
    plot_choice = ask_choice("\nQuale grafico vuoi generare?", choices=["1", "2"], default="1")

    if plot_choice == "1":
        console.print(f"\n[cyan]Generazione grafico a schermo: Convergenza Ottimizzatore...[/cyan]")
        plot_optimizer_convergence(
            sample_result['metrics']['optimization_history'], 
            f"Convergenza Ottimizzatore (N={n}, D={d:.2f}, p={p}, {opt})",
            filepath=None # Mostra a schermo invece di salvare
        )
    elif plot_choice == "2":
        if 'quasi_distribution' in sample_result['qaoa_results']:
            console.print(f"\n[cyan]Generazione grafico a schermo: Distribuzione di Probabilità...[/cyan]")
            plot_probability_distribution(
                sample_result['qaoa_results']['quasi_distribution'],
                title=f"Distribuzione di Probabilità (N={n}, D={d:.2f}, p={p}, {opt})",
                optimal_bitstring=sample_result['qaoa_results'].get('best_measured_bitstring'),
                filepath=None # Mostra a schermo invece di salvare
            )
        else:
            console.print("\n[bold red]Attenzione:[/bold red] I dati 'quasi_distribution' non sono stati salvati nel JSON per risparmiare spazio su disco.")
            console.print("Per visualizzare questo grafico, de-commenta la riga [italic]'quasi_distribution': qaoa_run_results['quasi_distribution'][/italic] in [green]src/qaoa/execute_benchmarking.py[/green] ed esegui nuovamente il benchmark.")

if __name__ == "__main__":
    main()
