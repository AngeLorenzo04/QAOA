import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.primitives import Sampler

from src.qaoa.qaoa_runner import QAOARunner
from src.qaoa.optimizer import calculate_maxcut_value

def compute_cut_landscape(graph, ansatz_circuit, sampler, steps=22):
    """
    Computes the expected cut value over a grid of gamma and beta values.
    Using steps=22 (484 evaluations) to ensure maximum fluidity during 3D rotation.
    """
    gamma_vals = np.linspace(0, 2 * np.pi, steps)
    beta_vals = np.linspace(0, 2 * np.pi, steps)
    gamma_grid, beta_grid = np.meshgrid(gamma_vals, beta_vals)
    cut_grid = np.zeros_like(gamma_grid)
    
    num_qubits = graph.number_of_nodes()
    total_evals = steps * steps
    print(f"Calculating expected cut landscape grid ({steps}x{steps} = {total_evals} evaluations)...")
    
    for i in range(steps):
        for j in range(steps):
            g = gamma_grid[i, j]
            b = beta_grid[i, j]
            # Bind parameters by name: g is gamma, b is beta
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

def plot_gradient_descent_trajectory():
    # 1. Create a simple 4-node cycle graph
    print("Creating a 4-node cycle graph...")
    graph = nx.cycle_graph(4)
    max_possible_cut = 4.0
    
    # 2. Instantiate the QAOA runner with p=1
    runner = QAOARunner(graph, p_value=1)
    
    # 3. Run QAOA using Custom Gradient Descent ('GD')
    gd_variant = input("Scegli la variante di Gradient Descent (vanilla, momentum, adam) [default: adam]: ").strip().lower()
    if gd_variant not in ["vanilla", "momentum", "adam"]:
        gd_variant = "adam"
    print(f"Running QAOA with Custom Gradient Descent ('GD' - {gd_variant})...")
    results = runner.run(
        max_optimization_iterations=25,
        optimizer_method='GD',
        tol=1e-4,
        gd_method=gd_variant
    )
    
    # Extract clean trajectory params
    trajectory_params = np.array(results['metrics']['trajectory_params'])
    
    # Wrap parameters to their periodic domains for robust plotting (columns: 0 is beta, 1 is gamma) to [0, 2*pi]
    trajectory_params[:, 0] = np.mod(trajectory_params[:, 0], 2 * np.pi)
    trajectory_params[:, 1] = np.mod(trajectory_params[:, 1], 2 * np.pi)
    
    # Calculate expected cut values along the clean trajectory
    sampler = Sampler()
    num_qubits = graph.number_of_nodes()
    trajectory_cuts = []
    
    print("Evaluating expected cut values along the clean trajectory...")
    for params in trajectory_params:
        # params[0] is beta, params[1] is gamma
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
            bitstring = format(state_int, f'0{num_qubits}b')
            exp_val += prob * calculate_maxcut_value(graph, bitstring)
        trajectory_cuts.append(exp_val)
    
    betas = trajectory_params[:, 0]
    gammas = trajectory_params[:, 1]
    
    # 4. Calculate the 2D expected cut landscape grid (coarser grid for lag-free 3D rendering)
    gamma_grid, beta_grid, cut_grid = compute_cut_landscape(
        graph, runner.ansatz_circuit, sampler, steps=22
    )
    
    # =========================================================================
    # WINDOW 1: Graph Partition, 2D Energy Density Landscape & Convergence Analysis
    # =========================================================================
    fig1, (ax1, ax2_contour, ax3) = plt.subplots(1, 3, figsize=(23, 7))
    fig1.canvas.manager.set_window_title('QAOA: Graph Solution, 2D Energy Landscape & Convergence Curves')
    
    # --- PANEL 1: Graph with Cut Visualization ---
    best_bitstring = results['best_measured_bitstring']
    partition = [int(char) for char in best_bitstring]
    
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ['#1f77b4' if partition[node] == 0 else '#ff7f0e' for node in graph.nodes()]
    
    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges():
        if partition[u] != partition[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))
            
    # Draw graph elements
    nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='#cccccc', width=2, style='--', ax=ax1)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='#2c3e50', width=5, ax=ax1)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000, edgecolors='black', linewidths=1.5, ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_size=14, font_color='white', font_weight='bold', ax=ax1)
    
    # Custom legend for Graph
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black', label='Set A (bit 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black', label='Set B (bit 1)'),
        Line2D([0], [0], color='#2c3e50', lw=4, label='Cut Edges (Separated)'),
        Line2D([0], [0], color='#cccccc', lw=2, linestyle='--', label='Uncut Edges')
    ]
    ax1.legend(handles=legend_elements, loc='lower center', fontsize=11, frameon=True, shadow=True)
    ax1.set_title(f"1. Problem Graph & Max Cut Solution\nBest Partition: {best_bitstring} (Cut Value: {results['best_measured_cut_value']})\n"
                  f"Optimal Parameters: $\\gamma$ = {gammas[-1]:.4f}, $\\beta$ = {betas[-1]:.4f}", fontsize=13, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # --- PANEL 2: 2D Energy Density (Contour Plot) with GD Trajectory ---
    contour = ax2_contour.contourf(gamma_grid, beta_grid, -cut_grid, levels=50, cmap='plasma')
    colorbar_2d = fig1.colorbar(contour, ax=ax2_contour, pad=0.05, shrink=0.8)
    colorbar_2d.set_label('Costo $-\\langle C \\rangle$', fontsize=11)
    
    # Overlay the gradient descent path
    ax2_contour.plot(gammas, betas, color='#00ffff', linestyle='-', linewidth=2.5, label='Percorso GD', zorder=10)
    # Step dots
    ax2_contour.scatter(gammas, betas, color='white', edgecolor='black', s=25, zorder=11)
    # Highlight start and end points
    ax2_contour.scatter(gammas[0], betas[0], color='#2ecc71', marker='o', s=100, edgecolors='black', label='Inizio (Casuale)', zorder=12)
    ax2_contour.scatter(gammas[-1], betas[-1], color='#e74c3c', marker='*', s=180, edgecolors='black', label='Fine (Ottimo)', zorder=12)
    
    # --- Rilevazione e Annotazione delle Valli nel grafico 2D ---
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
        
        # Evita sovrapposizione di etichette/indicatori
        too_close = False
        for lx, ly in labeled_coords:
            if np.sqrt((g - lx)**2 + (b - ly)**2) < 1.0:
                too_close = True
                break
        if not too_close:
            labeled_coords.append((g, b))
            sol = get_top_solutions_at_point(graph, runner.ansatz_circuit, sampler, g, b)
            
            # Distinzione tra minimo globale e locale
            if val < -3.5:
                marker_style = 'v'
                marker_color = 'red'
                marker_size = 70
                leg_key = f"Min Globale ({sol})"
            else:
                marker_style = 'o'
                marker_color = 'gold'
                marker_size = 55
                leg_key = f"Min Locale ({sol})"
            
            # Aggiungi alla legenda solo una volta per categoria
            leg_label = leg_key if leg_key not in legend_labels_added else None
            if leg_label:
                legend_labels_added.add(leg_key)
            
            # Disegna l'indicatore sulla mappa di calore 2D
            ax2_contour.scatter(g, b, marker=marker_style, color=marker_color, s=marker_size, 
                                edgecolors='black', label=leg_label, zorder=13)
            
            # Etichetta di testo con la bitstring
            ax2_contour.text(g + 0.12, b + 0.12, sol, fontsize=7, fontweight='bold', color='black',
                             bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='gray', lw=0.5),
                             zorder=14)
    
    ax2_contour.set_xlabel('$\\gamma$ (Costo)', fontsize=12)
    ax2_contour.set_ylabel('$\\beta$ (Mixer)', fontsize=12)
    ax2_contour.set_xlim(0, 2 * np.pi)
    ax2_contour.set_ylim(0, 2 * np.pi)
    ax2_contour.set_xticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
    ax2_contour.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
    ax2_contour.set_yticks(np.arange(0, 2 * np.pi + 0.1, np.pi / 2))
    ax2_contour.set_yticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
    ax2_contour.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=0.9, shadow=True)
    ax2_contour.grid(True, linestyle=':', alpha=0.5)
    ax2_contour.set_title("2. Densità di Energia e Percorso GD\n(Panorama 2D con traiettoria)", fontsize=13, fontweight='bold', pad=15)
    
    # --- PANEL 3: 1D Parameter and Cost (-Cut) evolution ---
    iterations = np.arange(len(trajectory_cuts))
    
    color = 'tab:blue'
    ax3.set_xlabel('Optimization Steps (Iterations)', fontsize=12)
    ax3.set_ylabel('Parameter Angle (radians)', color=color, fontsize=12)
    line1 = ax3.plot(iterations, gammas, label='$\\gamma$ (Cost Phase)', color='#1f77b4', marker='o', linestyle='-', linewidth=2)
    line2 = ax3.plot(iterations, betas, label='$\\beta$ (Mixer Phase)', color='#17becf', marker='s', linestyle='--', linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, linestyle=':', alpha=0.5)
    
    # Secondary y-axis for Negative Expected Cut (Cost to minimize)
    ax3_twin = ax3.twinx()
    color = 'tab:red'
    ax3_twin.set_ylabel('Valore del Costo $f(\\gamma, \\beta) = -\\langle C \\rangle$', color=color, fontsize=12)
    negative_trajectory_cuts = [-c for c in trajectory_cuts]
    line3 = ax3_twin.plot(iterations, negative_trajectory_cuts, label='Costo $f(\\gamma, \\beta)$', color='#d62728', marker='x', linestyle='-.', linewidth=2)
    ax3_twin.tick_params(axis='y', labelcolor=color)
    ax3_twin.set_ylim(-max_possible_cut - 0.2, 0.0)
    
    # Draw horizontal dashed line for theoretical Min Cost (-4.0)
    ax3_twin.axhline(y=-max_possible_cut, color='gray', linestyle=':', linewidth=1.5, label='Minimo Costo Teorico (-4.0)')
    
    # Legend handling for twin axes
    lines = line1 + line2 + line3 + [Line2D([0], [0], color='gray', linestyle=':', lw=1.5)]
    labels = [l.get_label() for l in lines[:-1]] + ['Minimo Costo Teorico (-4.0)']
    ax3.legend(lines, labels, loc='lower left', fontsize=11, frameon=True, shadow=True)
    ax3.set_title("3. Parameter angles & Cost convergence\n(Cost function $f$ descends towards minimum value of -4)", fontsize=13, fontweight='bold', pad=15)
    
    fig1.tight_layout()

    # =========================================================================
    # WINDOW 2: Dedicated FLUID 3D Expected Cut Landscape
    # =========================================================================
    fig2 = plt.figure(figsize=(10, 8.5))
    fig2.canvas.manager.set_window_title('QAOA: 3D Expected Cut Landscape and GD Trajectory')
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # Draw 3D surface (antialiased=False and stride=1 makes it extremely fluid to rotate)
    surf = ax2.plot_surface(gamma_grid, beta_grid, -cut_grid, cmap='plasma', alpha=0.85, 
                            edgecolor='none', rstride=1, cstride=1, antialiased=False, zorder=1)
    colorbar = fig2.colorbar(surf, ax=ax2, pad=0.1, shrink=0.6)
    colorbar.set_label('Valore del Costo $-\\langle C \\rangle$', fontsize=11)
    
    # 3D spline interpolation to make the trajectory follow the landscape perfectly
    from scipy.interpolate import RectBivariateSpline
    gamma_vals_1d = np.linspace(0, 2 * np.pi, len(gamma_grid))
    beta_vals_1d = np.linspace(0, 2 * np.pi, len(beta_grid))
    spline = RectBivariateSpline(beta_vals_1d, gamma_vals_1d, -cut_grid)
    
    # Calculate Z coordinates for the step dots on the surface
    trajectory_cuts_surface = np.array([spline(betas[k], gammas[k])[0, 0] for k in range(len(gammas))])
    
    z_offset = 0.04
    trajectory_cuts_offset = trajectory_cuts_surface + z_offset
    
    # Generate a finely interpolated path to make the 3D line follow the surface curvature
    fine_gammas = []
    fine_betas = []
    
    for i in range(len(gammas) - 1):
        g_start, g_end = gammas[i], gammas[i+1]
        b_start, b_end = betas[i], betas[i+1]
        
        # Interpolate 30 points per segment
        g_segment = np.linspace(g_start, g_end, 30)
        b_segment = np.linspace(b_start, b_end, 30)
        
        # If a boundary crossing is detected, insert a NaN to split the line (avoids drawing lines wrapping across the plot)
        if abs(g_end - g_start) > np.pi or abs(b_end - b_start) > np.pi:
            fine_gammas.extend([g_start, np.nan, g_end])
            fine_betas.extend([b_start, np.nan, b_end])
        else:
            if i > 0:
                fine_gammas.extend(g_segment[1:])
                fine_betas.extend(b_segment[1:])
            else:
                fine_gammas.extend(g_segment)
                fine_betas.extend(b_segment)
                
    fine_gammas = np.array(fine_gammas)
    fine_betas = np.array(fine_betas)
    
    fine_cuts_offset = []
    for idx in range(len(fine_gammas)):
        g_val = fine_gammas[idx]
        b_val = fine_betas[idx]
        if np.isnan(g_val) or np.isnan(b_val):
            fine_cuts_offset.append(np.nan)
        else:
            fine_cuts_offset.append(spline(b_val, g_val)[0, 0] + z_offset)
    fine_cuts_offset = np.array(fine_cuts_offset)
    
    # Plot the 3D trajectory line using the finely interpolated surface path
    ax2.plot(fine_gammas, fine_betas, fine_cuts_offset, color='#00ffff', linestyle='-', linewidth=2.5, label='Percorso Ottimizzazione', zorder=10)
    
    # Colored step dots (smaller, cleaner)
    step_colors = plt.cm.summer(np.linspace(0.0, 1.0, len(gammas)))
    ax2.scatter(gammas, betas, trajectory_cuts_offset, color=step_colors, s=30, edgecolors='black', linewidths=0.8, depthshade=False, zorder=11)
    
    # Highlight start and end points in 3D
    ax2.scatter(gammas[0], betas[0], trajectory_cuts_offset[0], color='#2ecc71', marker='o', s=100, edgecolors='black', label='Start (Casuale)', depthshade=False, zorder=12)
    ax2.scatter(gammas[-1], betas[-1], trajectory_cuts_offset[-1], color='#e74c3c', marker='*', s=200, edgecolors='black', label='End (Ottimo)', depthshade=False, zorder=12)
    
    # Label ONLY Start and End points cleanly in black (bold) for high contrast and readability against grid
    dist_start_end = np.sqrt((gammas[0] - gammas[-1])**2 + (betas[0] - betas[-1])**2)
    if dist_start_end < 1.2:
        # Shift labels in opposite directions to prevent overlap
        ax2.text(gammas[0] - 0.5, betas[0] - 0.5, trajectory_cuts_offset[0] + 0.15, 
                 f"Inizio\n$\\gamma$={gammas[0]:.3f}\n$\\beta$={betas[0]:.3f}", 
                 color='black', fontsize=10, weight='bold', zorder=13)
        ax2.text(gammas[-1] + 0.5, betas[-1] + 0.5, trajectory_cuts_offset[-1] + 0.25, 
                 f"Ottimo\n(Costo={-trajectory_cuts[-1]:.2f})\n$\\gamma$={gammas[-1]:.3f}\n$\\beta$={betas[-1]:.3f}", 
                 color='black', fontsize=10, weight='bold', zorder=13)
    else:
        # Standard offsets
        ax2.text(gammas[0] + 0.1, betas[0] + 0.1, trajectory_cuts_offset[0] + 0.15, 
                 f"Inizio\n$\\gamma$={gammas[0]:.3f}\n$\\beta$={betas[0]:.3f}", 
                 color='black', fontsize=10, weight='bold', zorder=13)
        ax2.text(gammas[-1] - 0.1, betas[-1] - 0.1, trajectory_cuts_offset[-1] + 0.25, 
                 f"Ottimo\n(Costo={-trajectory_cuts[-1]:.2f})\n$\\gamma$={gammas[-1]:.3f}\n$\\beta$={betas[-1]:.3f}", 
                 color='black', fontsize=10, weight='bold', zorder=13)
            
    ax2.set_title("2. Panorama 3D del Costo e Traiettoria GD\n(Il percorso scende verso il minimo)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("$\\gamma$ (Costo)", fontsize=11)
    ax2.set_ylabel("$\\beta$ (Mixer)", fontsize=11)
    ax2.set_zlabel("Costo $-\\langle C \\rangle$", fontsize=11)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_zlim(-max_possible_cut, 0)
    
    # Adjust initial view angle
    ax2.view_init(elev=35, azim=-60)
    ax2.legend(loc='upper left', fontsize=9, frameon=True, facecolor='black', framealpha=0.7, labelcolor='white')
    
    fig2.tight_layout()
    
    # Print exploration instructions in the terminal
    print("\n" + "="*60)
    print("ISTRUZIONI PER L'ESPLORAZIONE INTERATTIVA DEL GRAFICO 3D:")
    print("1. RUOTARE: Clicca col TASTO SINISTRO del mouse sul grafico 3D e trascina.")
    print("2. ZOOM: Clicca col TASTO DESTRO del mouse sul grafico 3D e trascina su/giù,")
    print("   oppure usa la ROTELLINA del mouse.")
    print("3. PAN (Spostare): Clicca premendo la ROTELLINA del mouse e trascina.")
    print("4. RIPRISTINO VIEW: Nella barra degli strumenti in basso alla finestra,")
    print("   clicca sull'icona della CASETTA per resettare la vista iniziale.")
    print("="*60 + "\n")
    
    # Open both interactive plot windows
    print("Opening interactive GUI plot windows...")
    plt.show()

if __name__ == "__main__":
    plot_gradient_descent_trajectory()
