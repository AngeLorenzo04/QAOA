import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from matplotlib.lines import Line2D

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.qaoa.optimizer import calculate_maxcut_value

def compute_cut_landscape(graph, ansatz_circuit, sampler, steps=22):
    """
    Computes expected cut values over a grid of gamma and beta values.
    """
    gamma_vals = np.linspace(0, 2 * np.pi, steps)
    beta_vals = np.linspace(0, 2 * np.pi, steps)
    gamma_grid, beta_grid = np.meshgrid(gamma_vals, beta_vals)
    cut_grid = np.zeros_like(gamma_grid)
    
    num_qubits = graph.number_of_nodes()
    
    for i in range(steps):
        for j in range(steps):
            g = gamma_grid[i, j]
            b = beta_grid[i, j]
            
            param_dict = {}
            for param in ansatz_circuit.parameters:
                if 'gamma' in param.name:
                    param_dict[param] = g
                elif 'beta' in param.name:
                    param_dict[param] = b
                    
            bound_circuit = ansatz_circuit.assign_parameters(param_dict)
            measured_circuit = bound_circuit.measure_all(inplace=False)
            
            job = sampler.run(measured_circuit, shots=8192)
            quasi_distribution = job.result().quasi_dists[0]
            
            exp_val = 0.0
            for state_int, prob in quasi_distribution.items():
                bitstring = format(state_int, f'0{num_qubits}b')
                exp_val += prob * calculate_maxcut_value(graph, bitstring)
                
            cut_grid[i, j] = exp_val
            
    return gamma_grid, beta_grid, cut_grid

def get_top_solutions_at_point(graph, ansatz_circuit, sampler, g, b):
    param_dict = {}
    for param in ansatz_circuit.parameters:
        if 'gamma' in param.name:
            param_dict[param] = g
        elif 'beta' in param.name:
            param_dict[param] = b
    bound_circuit = ansatz_circuit.assign_parameters(param_dict)
    num_qubits = graph.number_of_nodes()
    
    if num_qubits <= 12:
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(bound_circuit)
        quasi_distribution = {i: prob for i, prob in enumerate(sv.probabilities()) if prob > 1e-6}
    else:
        measured_circuit = bound_circuit.measure_all(inplace=False)
        job = sampler.run(measured_circuit, shots=16384)
        quasi_distribution = job.result().quasi_dists[0]
    
    sorted_outcomes = sorted(quasi_distribution.items(), key=lambda item: item[1], reverse=True)
    
    # Filter states with probability close to the maximum
    max_prob = sorted_outcomes[0][1]
    top_outcomes = []
    for state_int, prob in sorted_outcomes:
        if prob >= max_prob * 0.8:
            bitstring = format(state_int, f'0{num_qubits}b')
            top_outcomes.append(bitstring)
    return "/".join(top_outcomes)


class PlotGradientDescentPlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "plot_gradient_descent"

    @property
    def description(self) -> str:
        return "Esegui Gradient Descent e traccia la traiettoria sul panorama energetico"

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()
        
        if n_nodes >= 16:
            console.print("[bold red]Attenzione: N >= 16 richiede troppa memoria per la simulazione e la discesa del gradiente quantistica.[/bold red]")
            proceed = Prompt.ask("Vuoi procedere lo stesso?", choices=["si", "no"], default="no")
            if proceed.lower() != "si":
                return

        sampler = Sampler()
        
        # 1. Esegui QAOA con Gradient Descent ('GD')
        from rich.prompt import Prompt
        gd_variant = Prompt.ask("Scegli la variante di Gradient Descent", choices=["vanilla", "momentum", "adam"], default="adam")
        console.print(f"[green]Esecuzione di QAOA con Custom Gradient Descent ('GD' - {gd_variant})...[/green]")
        with console.status("[bold magenta]Ottimizzazione dei parametri in corso...[/bold magenta]"):
            results = runner.run(
                max_optimization_iterations=25,
                optimizer_method='GD',
                tol=1e-4,
                gd_method=gd_variant
            )
            
        trajectory_params = np.array(results['metrics']['trajectory_params'])
        trajectory_params[:, 0] = np.mod(trajectory_params[:, 0], 2 * np.pi)
        trajectory_params[:, 1] = np.mod(trajectory_params[:, 1], 2 * np.pi)
        
        trajectory_cuts = []
        with console.status("[bold yellow]Valutazione del costo lungo la traiettoria...[/bold yellow]"):
            for params in trajectory_params:
                param_dict = {}
                for param in runner.ansatz_circuit.parameters:
                    if 'beta' in param.name:
                        param_dict[param] = params[0]
                    elif 'gamma' in param.name:
                        param_dict[param] = params[1]
                bound_circuit = runner.ansatz_circuit.assign_parameters(param_dict)
                measured_circuit = bound_circuit.measure_all(inplace=False)
                job = sampler.run(measured_circuit, shots=16384)
                quasi_distribution = job.result().quasi_dists[0]
                
                exp_val = 0.0
                for state_int, prob in quasi_distribution.items():
                    bitstring = format(state_int, f'0{n_nodes}b')
                    exp_val += prob * calculate_maxcut_value(graph, bitstring)
                trajectory_cuts.append(exp_val)
                
        betas = trajectory_params[:, 0]
        gammas = trajectory_params[:, 1]
        
        with console.status("[bold cyan]Calcolo del panorama 2D di confronto (22x22 grid)...[/bold cyan]"):
            gamma_grid, beta_grid, cut_grid = compute_cut_landscape(graph, runner.ansatz_circuit, sampler, steps=22)
            
        # =========================================================================
        # WINDOW 1: Graph Partition, 2D Energy Density Landscape & Convergence Analysis
        # =========================================================================
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
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black', label='Set A (bit 0)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black', label='Set B (bit 1)'),
            Line2D([0], [0], color='#2c3e50', lw=4, label='Taglio (Separati)'),
            Line2D([0], [0], color='#cccccc', lw=2, linestyle='--', label='Non Taglio')
        ]
        ax1.legend(handles=legend_elements, loc='lower center', fontsize=11, frameon=True, shadow=True)
        ax1.set_title(f"1. Grafo e Taglio Massimo\nMigliore Partizione: {best_bitstring} (Costo: {results['best_measured_cut_value']})\n"
                      f"Parametri Ottimi: $\\gamma$ = {gammas[-1]:.4f}, $\\beta$ = {betas[-1]:.4f}", fontsize=13, fontweight='bold', pad=15)
        ax1.axis('off')
        
        # Panel 2: Heatmap 2D
        contour = ax2_contour.contourf(gamma_grid, beta_grid, -cut_grid, levels=50, cmap='plasma')
        colorbar_2d = fig1.colorbar(contour, ax=ax2_contour, pad=0.05, shrink=0.8)
        colorbar_2d.set_label('Costo $-\\langle C \\rangle$', fontsize=11)
        
        ax2_contour.plot(gammas, betas, color='#00ffff', linestyle='-', linewidth=2.5, label='Percorso GD', zorder=10)
        ax2_contour.scatter(gammas, betas, color='white', edgecolor='black', s=25, zorder=11)
        ax2_contour.scatter(gammas[0], betas[0], color='#2ecc71', marker='o', s=100, edgecolors='black', label='Inizio (Casuale)', zorder=12)
        ax2_contour.scatter(gammas[-1], betas[-1], color='#e74c3c', marker='*', s=180, edgecolors='black', label='Fine (Ottimo)', zorder=12)
        
        # Valley detection in GD plot
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
        iterations = range(len(trajectory_cuts))
        ax3.plot(iterations, -np.array(trajectory_cuts), color='#e74c3c', marker='o', linestyle='-', linewidth=2, label='Costo Atteso $-\\langle C \\rangle$')
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
        
        # Overlay GD trajectory in 3D
        ax_3d.plot(gammas, betas, -np.array(trajectory_cuts), color='#00ffff', linewidth=4, label='Traiettoria GD', zorder=10)
        ax_3d.scatter(gammas[0], betas[0], -trajectory_cuts[0], color='#2ecc71', marker='o', s=120, edgecolors='black', label='Inizio', zorder=11)
        ax_3d.scatter(gammas[-1], betas[-1], -trajectory_cuts[-1], color='#e74c3c', marker='*', s=250, edgecolors='black', label='Ottimo', zorder=11)
        
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
