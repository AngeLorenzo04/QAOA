import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.primitives import Sampler

from src.qaoa.qaoa_runner import QAOARunner
from src.qaoa.optimizer import calculate_maxcut_value

def compute_energy_landscape(graph, ansatz_circuit, sampler, steps=40):
    """
    Computes the QAOA energy expectation value over a grid of gamma and beta values.
    """
    gamma_vals = np.linspace(0, 2 * np.pi, steps)
    beta_vals = np.linspace(0, np.pi, steps)
    gamma_grid, beta_grid = np.meshgrid(gamma_vals, beta_vals)
    energy_grid = np.zeros_like(gamma_grid)
    
    num_qubits = graph.number_of_nodes()
    total_evals = steps * steps
    print(f"Calculating energy landscape grid ({steps}x{steps} = {total_evals} evaluations)...")
    
    for i in range(steps):
        for j in range(steps):
            g = gamma_grid[i, j]
            b = beta_grid[i, j]
            params = [g, b]
            bound_circuit = ansatz_circuit.assign_parameters(dict(zip(ansatz_circuit.parameters, params)))
            measured_circuit = bound_circuit.measure_all(inplace=False)
            job = sampler.run(measured_circuit, shots=1024)
            quasi_distribution = job.result().quasi_dists[0]
            
            exp_val = 0.0
            for state_int, prob in quasi_distribution.items():
                bitstring = format(state_int, f'0{num_qubits}b')
                exp_val += prob * calculate_maxcut_value(graph, bitstring)
            # Energy to minimize is the negative expected value
            energy_grid[i, j] = -exp_val
            
    return gamma_grid, beta_grid, energy_grid

def plot_gradient_descent_trajectory():
    # 1. Create a simple 4-node cycle graph
    print("Creating a 4-node cycle graph...")
    graph = nx.cycle_graph(4)
    
    # 2. Instantiate the QAOA runner with p=1
    runner = QAOARunner(graph, p_value=1)
    
    # 3. Run QAOA using Custom Gradient Descent ('GD')
    print("Running QAOA with Custom Gradient Descent ('GD')...")
    # Using 40 max iterations, learning_rate=0.1
    results = runner.run(
        max_optimization_iterations=40,
        optimizer_method='GD',
        tol=1e-4
    )
    
    # Extract clean trajectory params
    trajectory_params = np.array(results['metrics']['trajectory_params'])
    
    # Calculate energy values for each trajectory parameter set
    sampler = Sampler()
    num_qubits = graph.number_of_nodes()
    trajectory_fun = []
    
    print("Evaluating energy values along the clean trajectory...")
    for params in trajectory_params:
        bound_circuit = runner.ansatz_circuit.assign_parameters(dict(zip(runner.ansatz_circuit.parameters, params)))
        measured_circuit = bound_circuit.measure_all(inplace=False)
        job = sampler.run(measured_circuit, shots=1024)
        quasi_distribution = job.result().quasi_dists[0]
        
        exp_val = 0.0
        for state_int, prob in quasi_distribution.items():
            bitstring = format(state_int, f'0{num_qubits}b')
            exp_val += prob * calculate_maxcut_value(graph, bitstring)
        trajectory_fun.append(-exp_val)
    
    gammas = trajectory_params[:, 0]
    betas = trajectory_params[:, 1]
    
    # 4. Calculate the 2D energy landscape grid
    gamma_grid, beta_grid, energy_grid = compute_energy_landscape(
        graph, runner.ansatz_circuit, sampler, steps=30
    )
    
    # 5. Create a 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    
    # --- PANEL 1: Graph with Cut Visualization ---
    best_bitstring = results['best_measured_bitstring']
    partition = [int(char) for char in best_bitstring]
    
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ['skyblue' if partition[node] == 0 else 'salmon' for node in graph.nodes()]
    
    cut_edges = []
    non_cut_edges = []
    for u, v in graph.edges():
        if partition[u] != partition[v]:
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))
            
    # Draw graph elements
    nx.draw_networkx_edges(graph, pos, edgelist=non_cut_edges, edge_color='lightgray', width=2, ax=ax1)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color='black', width=4, ax=ax1)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, edgecolors='black', ax=ax1)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='black', ax=ax1)
    
    ax1.set_title(f"Cycle Graph (N=4) with Max Cut Partition\nBest Bitstring: {best_bitstring} (Cut: {results['best_measured_cut_value']})", fontsize=13)
    ax1.axis('off')
    
    # --- PANEL 2: 2D energy landscape contour & clean trajectory path ---
    contour = ax2.contourf(gamma_grid, beta_grid, energy_grid, levels=25, cmap='viridis')
    fig.colorbar(contour, ax=ax2, label='Energy Cost (Negative MaxCut Expectation)')
    
    # Plot trajectory path
    ax2.plot(gammas, betas, color='white', linestyle='-', linewidth=2.5, label='GD Optimization Path', zorder=2)
    ax2.scatter(gammas, betas, color='cyan', s=40, edgecolors='black', zorder=3)
    
    # Highlight start and end points
    ax2.scatter(gammas[0], betas[0], color='green', marker='o', s=150, edgecolors='black', label='Start (Initial)', zorder=4)
    ax2.scatter(gammas[-1], betas[-1], color='red', marker='*', s=250, edgecolors='black', label='End (Optimal)', zorder=4)
    
    # Add direction arrows along the path
    for k in range(len(gammas) - 1):
        ax2.annotate('', xy=(gammas[k+1], betas[k+1]), xytext=(gammas[k], betas[k]),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.5, mutation_scale=15), zorder=3)
        
    ax2.set_title("2D Energy Landscape and GD Trajectory Path", fontsize=13)
    ax2.set_xlabel("$\\gamma$ (Gamma Parameter)", fontsize=11)
    ax2.set_ylabel("$\\beta$ (Beta Parameter)", fontsize=11)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(0, np.pi)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- PANEL 3: 1D Parameter and Cost function evolution ---
    iterations = np.arange(len(trajectory_fun))
    
    color = 'tab:blue'
    ax3.set_xlabel('Optimization Steps (Iterations)', fontsize=11)
    ax3.set_ylabel('Parameter Angle (radians)', color=color, fontsize=11)
    line1 = ax3.plot(iterations, gammas, label='$\\gamma$', color='blue', marker='o', linestyle='-')
    line2 = ax3.plot(iterations, betas, label='$\\beta$', color='cyan', marker='s', linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # Secondary y-axis for energy value
    ax3_twin = ax3.twinx()
    color = 'tab:red'
    ax3_twin.set_ylabel('Objective Value (Energy Cost)', color=color, fontsize=11)
    line3 = ax3_twin.plot(iterations, trajectory_fun, label='Energy Cost', color='red', marker='x', linestyle='-.')
    ax3_twin.tick_params(axis='y', labelcolor=color)
    
    # Legend handling for twin axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.set_title("Parameter & Energy Convergence (Clean Steps)", fontsize=13)
    
    plt.tight_layout()
    
    # Save the output image
    output_filename = "gradient_descent_trajectory.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nPlot saved successfully as '{output_filename}' in the root directory!")
    
    plt.show()

if __name__ == "__main__":
    plot_gradient_descent_trajectory()
