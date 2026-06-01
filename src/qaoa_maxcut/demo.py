import pennylane as qml
from pennylane import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

from qaoa_maxcut.graph_definition import create_cycle_graph
from qaoa_maxcut.plotting import plot_dashboard
from qaoa_maxcut.qaoa_components import build_maxcut_hamiltonians
from qaoa_maxcut.circuit import create_qaoa_circuit, create_sampling_circuit


def main() -> None:
    console = Console()
    console.rule("[bold blue]QAOA Max-Cut Demonstration[/bold blue]")

    # 1. Define the graph
    graph = create_cycle_graph()
    n_wires = len(graph.nodes)

    graph_info = f"Nodes: {list(graph.nodes())}\nEdges: {list(graph.edges())}\nTotal Qubits: {n_wires}"
    console.print(Panel(graph_info, title="[bold green]Graph Definition[/bold green]", expand=False))

    # 2. Build Hamiltonians
    cost_h, mixer_h = build_maxcut_hamiltonians(graph)

    # 3. Create circuits
    circuit = create_qaoa_circuit(graph, cost_h, mixer_h)

    # 4. Set parameters
    p = 2
    np.random.seed(42)
    gammas = np.random.uniform(0, np.pi, p, requires_grad=True)
    betas = np.random.uniform(0, np.pi, p, requires_grad=True)
    params = np.array([gammas, betas], requires_grad=True)

    console.print(f"[cyan]Initialized parameters for p={p} layers.[/cyan]\n")

    # 5. Optimization Loop with Rich Progress Bar
    opt = qml.AdagradOptimizer(stepsize=0.5)
    steps = 40
    cost_history = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[magenta]Optimizing QAOA Circuit...", total=steps)
        
        for i in range(steps):
            params = opt.step(circuit, params)
            current_cost = circuit(params)
            cost_history.append(current_cost)
            
            # Update the progress bar with the current cost
            progress.update(task, advance=1, description=f"[magenta]Optimizing...[/magenta] (Cost: {current_cost:.4f})")

    console.print("\n[bold green]✔ Optimization Complete![/bold green]")

    # 6. Sampling
    sampling_circuit = create_sampling_circuit(graph, cost_h, mixer_h)
    probs = sampling_circuit(params)
    
    # 7. Find Best Result
    best_idx = np.argmax(probs)
    best_bitstring = format(best_idx, f'0{n_wires}b')

    result_info = f"Optimal Bitstring: [bold yellow]{best_bitstring}[/bold yellow]\nProbability: {probs[best_idx]:.4f}\nFinal Cost: {cost_history[-1]:.4f}"
    console.print(Panel(result_info, title="[bold green]Results[/bold green]", expand=False))

    # 8. Launch Unified Plotting Dashboard
    console.print("[dim]Launching unified plotting dashboard... Close the window to exit.[/dim]")
    plot_dashboard(graph, probs, best_bitstring, cost_history, n_wires)


if __name__ == "__main__":
    main()
