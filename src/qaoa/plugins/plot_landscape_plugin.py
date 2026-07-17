import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.qaoa.optimizer import calculate_maxcut_value

def get_canonical_partition(bitstring):
    comp = "".join('1' if c == '0' else '0' for c in bitstring)
    return min(bitstring, comp)

def compute_cut_landscape(graph, ansatz_circuit, sampler, steps=25):
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
            g_val = gamma_grid[i, j]
            b_val = beta_grid[i, j]
            
            param_dict = {}
            for param in ansatz_circuit.parameters:
                if 'gamma' in param.name:
                    param_dict[param] = g_val
                elif 'beta' in param.name:
                    param_dict[param] = b_val
                    
            bound_circuit = ansatz_circuit.assign_parameters(param_dict)
            measured_circuit = bound_circuit.measure_all(inplace=False)
            
            job = sampler.run(measured_circuit, shots=4096)
            quasi_distribution = job.result().quasi_dists[0]
            
            exp_val = 0.0
            for state_int, prob in quasi_distribution.items():
                bitstring = format(state_int, f'0{num_qubits}b')[::-1]
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
    
    max_prob = sorted_outcomes[0][1]
    top_outcomes = []
    for state_int, prob in sorted_outcomes:
        if prob >= max_prob * 0.8:
            bitstring = format(state_int, f'0{num_qubits}b')[::-1]
            top_outcomes.append(bitstring)
    return "/".join(top_outcomes), sorted_outcomes[0][0], sorted_outcomes

def draw_graph_partition(ax, graph, bitstring_int, title):
    num_qubits = graph.number_of_nodes()
    bitstring = format(bitstring_int, f'0{num_qubits}b')[::-1]
    partition = [int(char) for char in bitstring]
    
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ['#1f77b4' if partition[node] == 0 else '#ff7f0e' for node in graph.nodes()]
    
    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges():
        if partition[u] != partition[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))
            
    labels = {node: f"{node}:{partition[node]}" for node in graph.nodes()}
            
    nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='#cccccc', width=2, style='--', ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='#2c3e50', width=4, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=600, edgecolors='black', linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color='white', font_weight='bold', ax=ax)
    
    title = f"{title}\nRami tagliati: {len(cut_edges)}"
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.axis('off')


class PlotLandscapePlugin(QAOACommandPlugin):
    @property
    def name(self) -> str:
        return "plot_landscape"

    @property
    def description(self) -> str:
        return "Visualizza il panorama energetico 2D con istogramma e valli delle soluzioni"

    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        graph = graph_info['graph']
        n_nodes = graph.number_of_nodes()
        
        if n_nodes >= 16:
            console.print("[bold red]Attenzione: N >= 16 richiede troppa memoria per calcolare lo Statevector o simulare il panorama.[/bold red]")
            proceed = Prompt.ask("Vuoi procedere lo stesso?", choices=["si", "no"], default="no")
            if proceed.lower() != "si":
                return

        sampler = Sampler()
        
        with console.status("[bold magenta]Calcolo del panorama 2D della densità di energia (25x25 grid)...[/bold magenta]"):
            gamma_grid, beta_grid, cut_grid = compute_cut_landscape(graph, runner.ansatz_circuit, sampler, steps=25)
            
        z_data = -cut_grid
        h, w = z_data.shape
        minima_points = []
        
        # Valley detection
        for i in range(1, h-1):
            for j in range(1, w-1):
                val = z_data[i, j]
                neighbors = [
                    z_data[i-1, j-1], z_data[i-1, j], z_data[i-1, j+1],
                    z_data[i, j-1],                   z_data[i, j+1],
                    z_data[i+1, j-1], z_data[i+1, j], z_data[i+1, j+1]
                ]
                if val < min(neighbors):
                    minima_points.append((i, j, val))
                    
        minima_points.sort(key=lambda x: x[2])
        
        all_minima = []
        unique_partitions = {}
        labeled_coords = []
        
        with console.status("[bold yellow]Analisi delle soluzioni nelle valli individuate...[/bold yellow]"):
            for i, j, val in minima_points:
                g = gamma_grid[i, j]
                b = beta_grid[i, j]
                
                too_close = False
                for lx, ly in labeled_coords:
                    if np.sqrt((g - lx)**2 + (b - ly)**2) < 0.8:
                        too_close = True
                        break
                if too_close:
                    continue
                    
                labeled_coords.append((g, b))
                sol_text, sol_int, sorted_outcomes = get_top_solutions_at_point(graph, runner.ansatz_circuit, sampler, g, b)
                
                num_qubits = graph.number_of_nodes()
                bitstring = format(sol_int, f'0{num_qubits}b')[::-1]
                canonical = get_canonical_partition(bitstring)
                
                minimum_info = {
                    'coords': (g, b),
                    'val': val,
                    'sol_text': sol_text,
                    'sol_int': sol_int,
                    'canonical': canonical,
                    'sorted_outcomes': sorted_outcomes
                }
                all_minima.append(minimum_info)
                
                if canonical not in unique_partitions:
                    unique_partitions[canonical] = minimum_info
                    
        unique_keys = list(unique_partitions.keys())[:3]
        num_sols = len(unique_keys)
        
        markers_pool = [
            ('▼', 'v', 'red', 'Min Globale'),
            ('●', 'o', 'gold', 'Min Locale 1'),
            ('■', 's', 'green', 'Min Locale 2'),
            ('◆', 'D', 'magenta', 'Min Locale 3')
        ]
        partition_markers = {}
        for idx, key in enumerate(unique_keys):
            partition_markers[key] = markers_pool[idx]
            
        import matplotlib.gridspec as gridspec
        
        num_cols = num_sols if num_sols > 0 else 1
        fig = plt.figure(figsize=(3 * num_cols + 4, 9.5))
        fig.canvas.manager.set_window_title(f"QAOA Landscape (N={graph_info['n_vertices']}, ID={graph_info['id']})")
        
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1.0], hspace=0.6)
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], width_ratios=[1.15, 1.0], wspace=0.35)
        ax_main = fig.add_subplot(gs_top[0, 0])
        ax_hist = fig.add_subplot(gs_top[0, 1])
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, num_cols, subplot_spec=gs_main[1], wspace=0.4)
            
        # 1. 2D Contour
        contour = ax_main.contourf(gamma_grid, beta_grid, -cut_grid, levels=50, cmap='plasma')
        colorbar = fig.colorbar(contour, ax=ax_main, pad=0.05, shrink=0.8)
        colorbar.set_label('Costo $-\\langle C \\rangle$', fontsize=11)
        
        legend_labels_added = set()
        for minimum in all_minima:
            g, b = minimum['coords']
            canonical = minimum['canonical']
            sol_text = minimum['sol_text']
            
            if canonical in partition_markers:
                unicode_char, marker_style, color, name = partition_markers[canonical]
            else:
                unicode_char, marker_style, color, name = ('▲', '^', 'gray', 'Altro Minimo')
                
            leg_label = f"{unicode_char} {name}"
            if leg_label not in legend_labels_added:
                legend_labels_added.add(leg_label)
                ax_main.scatter(g, b, marker=marker_style, color=color, s=90, edgecolors='black', label=leg_label, zorder=10)
            else:
                ax_main.scatter(g, b, marker=marker_style, color=color, s=90, edgecolors='black', zorder=10)
            
            ax_main.text(g + 0.12, b + 0.12, sol_text, fontsize=8, fontweight='bold', color='black',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='gray', lw=0.5),
                         zorder=11)
            
        ax_main.set_xlabel('$\\gamma$ (Costo)', fontsize=11)
        ax_main.set_ylabel('$\\beta$ (Mixer)', fontsize=11)
        ax_main.set_xlim(0, 2 * np.pi)
        ax_main.set_ylim(0, 2 * np.pi)
        ax_main.set_xticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax_main.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax_main.set_yticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
        ax_main.set_yticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
        ax_main.grid(True, linestyle=':', alpha=0.5)
        
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_labels_added),
                       frameon=True, shadow=True, fontsize=9)
        ax_main.set_title(f"Densità di Energia (N={graph_info['n_vertices']}, ID={graph_info['id']})", fontsize=12, fontweight='bold', pad=15)
        
        # 2. Histogram
        if num_sols > 0:
            global_min_key = unique_keys[0]
            global_min_data = unique_partitions[global_min_key]
            sorted_outcomes = global_min_data['sorted_outcomes']
            
            top_k = min(10, len(sorted_outcomes))
            x_labels = []
            y_probs = []
            for state_int, prob in sorted_outcomes[:top_k]:
                bitstring = format(state_int, f'0{n_nodes}b')[::-1]
                x_labels.append(bitstring)
                y_probs.append(prob)
                
            colors = []
            for bs in x_labels:
                canonical = get_canonical_partition(bs)
                if canonical in partition_markers:
                    colors.append(partition_markers[canonical][2])
                else:
                    colors.append('#3498db')
                    
            bars = ax_hist.bar(x_labels, y_probs, color=colors, edgecolor='black', width=0.6)
            
            for bar in bars:
                height = bar.get_height()
                ax_hist.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f"{height:.2f}",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
                             
            ax_hist.set_xticks(range(len(x_labels)))
            ax_hist.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax_hist.set_ylabel('Probabilità', fontsize=11)
            ax_hist.set_ylim(0, max(y_probs) * 1.15)
            ax_hist.set_title(f"Istogramma Probabilità al Min Globale (▼)\n(Top {top_k} Stati)", fontsize=11, fontweight='bold', pad=15)
            ax_hist.grid(axis='y', linestyle=':', alpha=0.6)
        
        # 3. Bottom Partition Drawings
        for idx, key in enumerate(unique_keys):
            ax_sol = fig.add_subplot(gs_bottom[0, idx])
            data = unique_partitions[key]
            unicode_char, marker_style, color, name = partition_markers[key]
            
            title = f"{unicode_char} {name}\nStato: {data['sol_text']}\nCosto: {data['val']:.2f}"
            draw_graph_partition(ax_sol, graph, data['sol_int'], title)
            
        fig.tight_layout()
        console.print("Visualizzazione del grafico in corso...")
        plt.show()
