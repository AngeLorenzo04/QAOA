import sys
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time

from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.data.exact_maxcut_solver import find_exact_maxcut, find_exact_maxcut_ilp
from src.common.plotting import plot_qaoa_dashboard
from src.qaoa.optimizer import calculate_maxcut_value

class RunQAOAPlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "run_qaoa"

    @property
    def description(self) -> str:
        return "Analisi classica ed esecuzione quantistica QAOA (COBYLA/GD)"

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        # 1. Scelta del tipo di analisi (Classica o Quantistica QAOA)
        console.print("\n[bold cyan]Scegli il tipo di analisi da eseguire:[/bold cyan]")
        console.print("  [yellow]1[/yellow] - Analisi Classica (Risolutore Esatto Max-Cut)")
        console.print("  [yellow]2[/yellow] - QAOA (Ottimizzazione Quantistica)")
        analysis_choice = Prompt.ask("Scegli un'opzione", choices=["1", "2"], default="2")
        analysis_type = "classico" if analysis_choice == "1" else "qaoa"
        
        if analysis_type == "classico":
            # --- ANALISI CLASSICA ---
            console.print("[green]Esecuzione del Risolutore Classico Esatto...[/green]")
            start_time = time.time()
            with console.status("[bold yellow]Calcolo in corso...[/bold yellow]"):
                if n_nodes <= 16:
                    exact_res = find_exact_maxcut(graph)
                else:
                    exact_res = find_exact_maxcut_ilp(graph)
            elapsed_time = time.time() - start_time
            
            exact_maxcut_val = exact_res['max_cut_value']
            partitions = exact_res.get('max_cut_partitions', [])
            if partitions:
                best_partition = partitions[0]
                best_bitstring = "".join(str(val) for val in best_partition)
                best_bitstrings = ["".join(str(val) for val in p) for p in partitions]
            else:
                best_bitstring = ''
                best_bitstrings = []
            
            # Salva il risultato nei metadati del grafo
            graph.graph['exact_max_cut_value'] = exact_maxcut_val
            
            result_text = (
                f"Taglio Massimo Esatto: {exact_maxcut_val}\n"
                f"Configurazioni Ottime ({len(best_bitstrings)} trovate): {', '.join(best_bitstrings[:5])}"
                f"{' ...' if len(best_bitstrings) > 5 else ''}\n"
                f"Tempo di Esecuzione: {elapsed_time:.4f} secondi"
            )
            console.print(Panel(result_text, title="[bold green]Risultati Analisi Classica[/bold green]", expand=False))
            
            plot_choice = Prompt.ask("Vuoi visualizzare il grafico della partizione classica?", choices=["si", "no"], default="si")
            if plot_choice.lower() in ["si", "s"]:
                self.plot_classical_partition(graph, exact_maxcut_val, best_bitstring)
                
        else:
            # --- ANALISI QUANTISTICA QAOA ---
            console.print("\n[bold cyan]=== CONFIGURAZIONE QAOA ===[/bold cyan]")
            p_layers = IntPrompt.ask("Inserisci il numero di layer QAOA (p)", default=runner.p_value)
            runner.update_p(p_layers)
            
            # 2. Scelta Ottimizzatore (COBYLA o GD)
            console.print("\n[bold cyan]Scegli l'ottimizzatore classico:[/bold cyan]")
            console.print("  [yellow]1[/yellow] - COBYLA")
            console.print("  [yellow]2[/yellow] - GD (Custom Gradient Descent)")
            opt_choice = Prompt.ask("Scegli un'opzione", choices=["1", "2"], default="1")
            optimizer = "COBYLA" if opt_choice == "1" else "GD"
            
            gd_variant = "adam"
            num_starts = 1
            if optimizer == 'GD':
                # Scelta del tipo di GD in caso di ottimizzatore GD
                console.print("\n[bold cyan]Scegli la variante di Gradient Descent:[/bold cyan]")
                console.print("  [yellow]1[/yellow] - Vanilla GD")
                console.print("  [yellow]2[/yellow] - Momentum GD")
                console.print("  [yellow]3[/yellow] - Adam (Default)")
                var_choice = Prompt.ask("Scegli un'opzione", choices=["1", "2", "3"], default="3")
                gd_variant = "vanilla" if var_choice == "1" else "momentum" if var_choice == "2" else "adam"
                
                num_starts = IntPrompt.ask("Scegli il numero di punti di partenza (Multi-start)", default=3)
            
            shots = IntPrompt.ask("Scegli il numero di shots", default=1024)
            
            # Calcola MaxCut classico di riferimento se non disponibile nei metadati del grafo
            exact_maxcut_val = graph.graph.get('exact_max_cut_value', -1)
            if exact_maxcut_val == -1:
                with console.status("[bold yellow]Calcolo del Max-Cut esatto classica per riferimento...[/bold yellow]"):
                    if n_nodes <= 16:
                        exact_res = find_exact_maxcut(graph)
                    else:
                        exact_res = find_exact_maxcut_ilp(graph)
                    exact_maxcut_val = exact_res['max_cut_value']
                    graph.graph['exact_max_cut_value'] = exact_maxcut_val
            
            # 3. Esegui QAOA
            with console.status(f"[bold magenta]Esecuzione di QAOA con {optimizer} (GD variant: {gd_variant if optimizer == 'GD' else 'N/A'}, starts: {num_starts})...[/bold magenta]"):
                qaoa_results = runner.run(
                    optimizer_method=optimizer,
                    shots=shots,
                    max_optimization_iterations=100 if optimizer != 'GD' else 25,
                    gd_method=gd_variant,
                    num_starts=num_starts
                )
                
            # 4. Mostra i risultati
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
            console.print(Panel(result_text, title="[bold green]Risultati Ottimizzazione QAOA[/bold green]", expand=False))
            
            # Mostra le top 5 configurazioni ordinate per probabilità
            quasi_dist = qaoa_results['quasi_distribution']
            sorted_quasi = sorted(quasi_dist.items(), key=lambda x: x[1], reverse=True)
            console.print("\n[bold cyan]Top 5 configurazioni misurate (ordinate per probabilità):[/bold cyan]")
            for bs, prob in sorted_quasi[:5]:
                cut_val = calculate_maxcut_value(graph, bs)
                console.print(f"  Stringa: [yellow]{bs}[/yellow] | Probabilità: [cyan]{prob:.4f}[/cyan] | Taglio: [green]{cut_val}[/green]")
            console.print("")
            
            # 5. Visualizzazione Grafici
            while True:
                if optimizer == 'GD':
                    console.print("\n[bold cyan]Scegli il grafico da visualizzare:[/bold cyan]")
                    console.print("  [yellow]1[/yellow] - Dashboard standard 1x3")
                    if runner.p_value == 1:
                        console.print("  [yellow]2[/yellow] - Traiettoria GD 2D/3D")
                        console.print("  [yellow]3[/yellow] - Nessuno")
                        plot_choice_num = Prompt.ask("Scegli un'opzione", choices=["1", "2", "3"], default="1")
                        plot_choice = "Dashboard standard 1x3" if plot_choice_num == "1" else "Traiettoria GD 2D/3D" if plot_choice_num == "2" else "Nessuno"
                    else:
                        console.print("  [yellow]2[/yellow] - Nessuno")
                        console.print("[yellow]Nota: La traiettoria GD 2D/3D è visualizzabile solo per p=1 layer quantistico (spazio di ricerca 2D).[/yellow]")
                        plot_choice_num = Prompt.ask("Scegli un'opzione", choices=["1", "2"], default="1")
                        plot_choice = "Dashboard standard 1x3" if plot_choice_num == "1" else "Nessuno"
                else:
                    plot_choice_str = Prompt.ask("Vuoi visualizzare la dashboard grafica dei risultati?", choices=["si", "no"], default="si")
                    plot_choice = "Dashboard standard 1x3" if plot_choice_str.lower() in ["si", "s"] else "Nessuno"
                    
                if plot_choice == "Nessuno":
                    break
                    
                if plot_choice == "Dashboard standard 1x3":
                    probs = np.zeros(2**n_nodes)
                    for bitstring, prob in quasi_dist.items():
                        state_int = int(bitstring, 2)
                        probs[state_int] = prob
                    plot_qaoa_dashboard(
                        graph=graph,
                        k=2,
                        probs=probs,
                        best_bitstring=best_bitstring,
                        cost_history=qaoa_results['metrics']['optimization_history'],
                        title=f"QAOA N={n_nodes}, D={graph_info['density_edges']:.2f}, ID={graph_info['id']} ({optimizer})"
                    )
                elif plot_choice == "Traiettoria GD 2D/3D":
                    self.plot_gd_trajectory(graph, runner, qaoa_results, console)

    def plot_classical_partition(self, graph: nx.Graph, exact_maxcut_val: int, best_bitstring: str) -> None:
        n_nodes = graph.number_of_nodes()
        sorted_nodes = sorted(graph.nodes())
        partition_dict = {node: int(best_bitstring[idx]) for idx, node in enumerate(sorted_nodes)}
        pos = nx.spring_layout(graph, seed=42)
        node_colors = ['#1f77b4' if partition_dict[node] == 0 else '#ff7f0e' for node in graph.nodes()]
        
        cut_edges = []
        non_cut_edges = []
        for u, v in graph.edges():
            if partition_dict[u] != partition_dict[v]:
                cut_edges.append((u, v))
            else:
                non_cut_edges.append((u, v))
                
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title('Risolutore Classico Max-Cut')
        
        nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='#cccccc', width=2, style='--', ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='#2c3e50', width=5, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000, edgecolors='black', linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=14, font_color='white', font_weight='bold', ax=ax)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black', label='Set A (bit 0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black', label='Set B (bit 1)'),
            Line2D([0], [0], color='#2c3e50', lw=4, label='Taglio (Separati)'),
            Line2D([0], [0], color='#cccccc', lw=2, linestyle='--', label='Non Taglio')
        ]
        ax.legend(handles=legend_elements, loc='lower center', fontsize=11, frameon=True, shadow=True)
        ax.set_title(f"Taglio Massimo Classico (Esatto)\nValore del Taglio: {exact_maxcut_val} (Rami tagliati: {len(cut_edges)}) | Soluzione: {best_bitstring}", fontsize=13, fontweight='bold', pad=15)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_gd_trajectory(self, graph: nx.Graph, runner: QAOARunner, results: dict, console) -> None:
        from src.qaoa.plugins.plot_gradient_descent_plugin import compute_cut_landscape, get_top_solutions_at_point
        from qiskit.primitives import Sampler
        
        n_nodes = graph.number_of_nodes()
        sampler = Sampler()
        
        all_trajectories = results['metrics'].get('all_trajectories', [results['metrics']['trajectory_params']])
        
        all_trajectory_cuts = []
        all_betas_list = []
        all_gammas_list = []
        
        p_val = runner.p_value
        with console.status("[bold yellow]Valutazione del costo lungo le traiettorie dei vari start...[/bold yellow]"):
            for traj in all_trajectories:
                traj_arr = np.array(traj)
                # Modulo 2pi per beta e gamma
                traj_arr[:, 0] = np.mod(traj_arr[:, 0], 2 * np.pi)
                traj_arr[:, 1] = np.mod(traj_arr[:, 1], 2 * np.pi)
                
                all_betas_list.append(traj_arr[:, 0])
                all_gammas_list.append(traj_arr[:, 1])
                
                cuts = []
                for params in traj_arr:
                    beta_params = params[:p_val]
                    gamma_params = params[p_val:]
                    
                    param_dict = {}
                    for param in runner.ansatz_circuit.parameters:
                        name = param.name
                        if 'beta' in name:
                            idx = int(name.split('[')[1].split(']')[0])
                            param_dict[param] = beta_params[idx]
                        elif 'gamma' in name:
                            idx = int(name.split('[')[1].split(']')[0])
                            param_dict[param] = gamma_params[idx]
                            
                    bound_circuit = runner.ansatz_circuit.assign_parameters(param_dict)
                    measured_circuit = bound_circuit.measure_all(inplace=False)
                    job = sampler.run(measured_circuit, shots=4096)
                    quasi_distribution = job.result().quasi_dists[0]
                    
                    exp_val = 0.0
                    for state_int, prob in quasi_distribution.items():
                        bitstring = format(state_int, f'0{n_nodes}b')[::-1]
                        exp_val += prob * calculate_maxcut_value(graph, bitstring)
                    cuts.append(exp_val)
                all_trajectory_cuts.append(cuts)
                
        # Trova la traiettoria migliore (quella che raggiunge il costo energetico minimo effettivo alla fine)
        best_idx = -1
        min_final_cost = float('inf')
        for idx, cuts in enumerate(all_trajectory_cuts):
            final_cost = -cuts[-1]
            if final_cost < min_final_cost:
                min_final_cost = final_cost
                best_idx = idx
                
        with console.status("[bold cyan]Calcolo del panorama 2D di confronto (22x22 grid)...[/bold cyan]"):
            gamma_grid, beta_grid, cut_grid = compute_cut_landscape(graph, runner.ansatz_circuit, sampler, steps=22)
            
        fig1, (ax1, ax2_contour, ax3) = plt.subplots(1, 3, figsize=(23, 7))
        fig1.canvas.manager.set_window_title('QAOA: Soluzione, Panorama 2D & Curve di Ottimizzazione')
        
        # Panel 1: Graph Partition
        best_bitstring = results['best_measured_bitstring']
        sorted_nodes = sorted(graph.nodes())
        partition_dict = {node: int(best_bitstring[idx]) for idx, node in enumerate(sorted_nodes)}
        
        pos = nx.spring_layout(graph, seed=42)
        node_colors = ['#1f77b4' if partition_dict[node] == 0 else '#ff7f0e' for node in graph.nodes()]
        
        cut_edges = []
        non_cut_edges = []
        for u, v in graph.edges():
            if partition_dict[u] != partition_dict[v]:
                cut_edges.append((u, v))
            else:
                non_cut_edges.append((u, v))
                
        nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='#cccccc', width=2, style='--', ax=ax1)
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='#2c3e50', width=5, ax=ax1)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000, edgecolors='black', linewidths=1.5, ax=ax1)
        nx.draw_networkx_labels(graph, pos, font_size=14, font_color='white', font_weight='bold', ax=ax1)
        
        best_gammas = all_gammas_list[best_idx]
        best_betas = all_betas_list[best_idx]
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black', label='Set A (bit 0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black', label='Set B (bit 1)'),
            Line2D([0], [0], color='#2c3e50', lw=4, label='Taglio (Separati)'),
            Line2D([0], [0], color='#cccccc', lw=2, linestyle='--', label='Non Taglio')
        ]
        ax1.legend(handles=legend_elements, loc='lower center', fontsize=11, frameon=True, shadow=True)
        exact_maxcut_val = graph.graph.get('exact_max_cut_value', -1)
        best_cuts = all_trajectory_cuts[best_idx]
        expected_cost = -best_cuts[-1]
        
        if exact_maxcut_val != -1 and exact_maxcut_val is not None and exact_maxcut_val > 0:
            ratio = len(cut_edges) / exact_maxcut_val
            ax1.set_title(f"1. Grafo e Taglio Massimo\nMigliore Partizione: {best_bitstring} (Taglio: {len(cut_edges)}/{exact_maxcut_val}, Ratio: {ratio:.4f}, Costo Atteso: {expected_cost:.4f})\n"
                          f"Parametri Ottimi: $\\gamma$ = {best_gammas[-1]:.4f}, $\\beta$ = {best_betas[-1]:.4f}", fontsize=13, fontweight='bold', pad=15)
        else:
            ax1.set_title(f"1. Grafo e Taglio Massimo\nMigliore Partizione: {best_bitstring} (Rami tagliati: {len(cut_edges)}, Costo Atteso: {expected_cost:.4f})\n"
                          f"Parametri Ottimi: $\\gamma$ = {best_gammas[-1]:.4f}, $\\beta$ = {best_betas[-1]:.4f}", fontsize=13, fontweight='bold', pad=15)
        ax1.axis('off')
        
        # Panel 2: Heatmap 2D
        contour = ax2_contour.contourf(gamma_grid, beta_grid, -cut_grid, levels=50, cmap='plasma')
        colorbar_2d = fig1.colorbar(contour, ax=ax2_contour, pad=0.05, shrink=0.8)
        colorbar_2d.set_label('Costo $-\\langle C \\rangle$', fontsize=11)
        
        # Disegna traiettorie alternative
        alternative_colors = ['#ff7675', '#a29bfe', '#74b9ff', '#ffeaa7', '#55efc4']
        alt_color_idx = 0
        for idx in range(len(all_trajectories)):
            if idx == best_idx:
                continue
            gammas_t = all_gammas_list[idx]
            betas_t = all_betas_list[idx]
            color = alternative_colors[alt_color_idx % len(alternative_colors)]
            alt_color_idx += 1
            
            ax2_contour.plot(gammas_t, betas_t, color=color, alpha=0.6, linestyle=':', linewidth=1.5, zorder=8)
            ax2_contour.scatter(gammas_t[0], betas_t[0], color=color, alpha=0.7, marker='o', s=60, edgecolors='black', zorder=9)
            ax2_contour.scatter(gammas_t[-1], betas_t[-1], color=color, alpha=0.7, marker='P', s=75, edgecolors='black', zorder=9)
            ax2_contour.scatter(gammas_t, betas_t, color='white', alpha=0.6, s=10, edgecolors='black', zorder=8)
            
        # Disegna la traiettoria migliore
        ax2_contour.plot(best_gammas, best_betas, color='#00ffff', label='GD (Migliore)', linewidth=3, zorder=10)
        ax2_contour.scatter(best_gammas, best_betas, color='white', edgecolor='black', s=25, zorder=11)
        ax2_contour.scatter(best_gammas[0], best_betas[0], color='#2ecc71', marker='o', s=160, edgecolors='black', label='Inizio (Migliore)', zorder=12)
        ax2_contour.scatter(best_gammas[-1], best_betas[-1], color='#e74c3c', marker='*', s=220, edgecolors='black', label='Fine (Migliore)', zorder=12)
        
        if len(all_trajectories) > 1:
            ax2_contour.plot([], [], color='gray', alpha=0.6, linestyle=':', linewidth=1.5, label='GD Alternativi')
            
        # Valley detection
        z_data = -cut_grid
        h, w = z_data.shape
        minima_points = []
        for i_coord in range(1, h-1):
            for j_coord in range(1, w-1):
                val = z_data[i_coord, j_coord]
                neighbors = [
                    z_data[i_coord-1, j_coord-1], z_data[i_coord-1, j_coord], z_data[i_coord-1, j_coord+1],
                    z_data[i_coord, j_coord-1],                               z_data[i_coord, j_coord+1],
                    z_data[i_coord+1, j_coord-1], z_data[i_coord+1, j_coord], z_data[i_coord+1, j_coord+1]
                ]
                if val < min(neighbors):
                    minima_points.append((i_coord, j_coord, val))
        minima_points.sort(key=lambda x: x[2])
        
        legend_labels_added = set()
        labeled_coords = []
        for i_coord, j_coord, val in minima_points:
            g = gamma_grid[i_coord, j_coord]
            b = beta_grid[i_coord, j_coord]
            
            too_close = False
            for lx, ly in labeled_coords:
                if np.sqrt((g - lx)**2 + (b - ly)**2) < 1.0:
                    too_close = True
                    break
            if not too_close:
                labeled_coords.append((g, b))
                
                # Check for global vs local
                if val < -0.8 * abs(np.min(z_data)):
                    marker_style = 'v'
                    marker_color = 'red'
                    marker_size = 70
                    leg_key = "Min Globale"
                else:
                    marker_style = 'o'
                    marker_color = 'gold'
                    marker_size = 55
                    leg_key = "Min Locale"
                
                leg_label = leg_key if leg_key not in legend_labels_added else None
                if leg_label:
                    legend_labels_added.add(leg_key)
                
                ax2_contour.scatter(g, b, marker=marker_style, color=marker_color, s=marker_size, 
                                    edgecolors='black', label=leg_label, zorder=13)
                                 
        ax2_contour.set_xlabel('$\\gamma$ (Costo)', fontsize=12)
        ax2_contour.set_ylabel('$\\beta$ (Mixer)', fontsize=12)
        ax2_contour.set_title("2. Mappa di Livello 2D dell'Energia\ne traiettoria del Gradient Descent", fontsize=13, fontweight='bold', pad=15)
        ax2_contour.set_xticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax2_contour.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax2_contour.set_yticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax2_contour.set_yticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax2_contour.grid(True, linestyle=':', alpha=0.5)
        ax2_contour.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=True, shadow=True, fontsize=9)
        
        # Panel 3: Convergence
        alt_color_idx = 0
        for idx, cuts in enumerate(all_trajectory_cuts):
            if idx == best_idx:
                continue
            iterations_t = range(len(cuts))
            color = alternative_colors[alt_color_idx % len(alternative_colors)]
            alt_color_idx += 1
            ax3.plot(iterations_t, -np.array(cuts), color=color, alpha=0.5, linestyle=':', linewidth=1.5)
            
        best_cuts = all_trajectory_cuts[best_idx]
        best_iterations = range(len(best_cuts))
        ax3.plot(best_iterations, -np.array(best_cuts), color='#e74c3c', marker='o', linestyle='-', linewidth=2.5, label='Costo Atteso (Migliore)')
        
        if len(all_trajectories) > 1:
            ax3.plot([], [], color='gray', alpha=0.5, linestyle=':', linewidth=1.5, label='GD Alternativi')
            
        ax3.axhline(y=-results['best_measured_cut_value'], color='black', linestyle='--', linewidth=1.5, label='Miglior Taglio Misurato')
        ax3.set_xlabel('Iterazioni di Gradient Descent', fontsize=12)
        ax3.set_ylabel('Valore del Taglio (Costo)', fontsize=12)
        ax3.set_title("3. Curva di Convergenza dell'Algoritmo", fontsize=13, fontweight='bold', pad=15)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
        
        fig1.tight_layout()
        
        # =========================================================================
        # WINDOW 2: 3D Surface expected cut landscape & Trajectory projection
        # =========================================================================
        fig2 = plt.figure(figsize=(12, 9))
        fig2.canvas.manager.set_window_title('QAOA: Traiettoria GD Proiettata in 3D')
        
        ax_3d = fig2.add_subplot(111, projection='3d')
        surf = ax_3d.plot_surface(gamma_grid, beta_grid, -cut_grid, cmap='plasma', alpha=0.8, edgecolor='none', zorder=1)
        
        # Disegni percorsi alternativi in 3D
        alt_color_idx = 0
        for idx, cuts in enumerate(all_trajectory_cuts):
            if idx == best_idx:
                continue
            gammas_t = all_gammas_list[idx]
            betas_t = all_betas_list[idx]
            color = alternative_colors[alt_color_idx % len(alternative_colors)]
            alt_color_idx += 1
            
            ax_3d.plot(gammas_t, betas_t, -np.array(cuts), color=color, alpha=0.5, linestyle=':', linewidth=1.5, zorder=8)
            ax_3d.scatter(gammas_t[0], betas_t[0], -cuts[0], color=color, alpha=0.6, marker='o', s=50, edgecolors='black', zorder=9)
            ax_3d.scatter(gammas_t[-1], betas_t[-1], -cuts[-1], color=color, alpha=0.6, marker='P', s=60, edgecolors='black', zorder=9)
            
        # Disegni percorso migliore in 3D
        best_cuts = all_trajectory_cuts[best_idx]
        ax_3d.plot(best_gammas, best_betas, -np.array(best_cuts), color='#00ffff', linewidth=4, label='Traiettoria GD (Migliore)', zorder=10)
        ax_3d.scatter(best_gammas[0], best_betas[0], -best_cuts[0], color='#2ecc71', marker='o', s=130, edgecolors='black', label='Inizio (Migliore)', zorder=11)
        ax_3d.scatter(best_gammas[-1], best_betas[-1], -best_cuts[-1], color='#e74c3c', marker='*', s=260, edgecolors='black', label='Ottimo (Migliore)', zorder=11)
        
        ax_3d.set_xlabel('$\\gamma$ (Costo)', fontsize=11, labelpad=10)
        ax_3d.set_ylabel('$\\beta$ (Mixer)', fontsize=11, labelpad=10)
        ax_3d.set_zlabel('Energia $\\langle H_C \\rangle$', fontsize=11, labelpad=10)
        ax_3d.set_title("Mappa Energetica 3D & Ottimizzazione QAOA", fontsize=14, fontweight='bold', pad=15)
        
        ax_3d.set_xticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax_3d.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax_3d.set_yticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax_3d.set_yticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        
        fig2.colorbar(surf, ax=ax_3d, pad=0.08, shrink=0.6, label='Costo')
        ax_3d.legend(loc='upper right', fontsize=10)
        
        console.print("Visualizzazione dei grafici in corso...")
        plt.show()
