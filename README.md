# QAOA Max-Cut Demonstration

This repository contains a complete demonstration of solving the Max-Cut problem using the Quantum Approximate Optimization Algorithm (QAOA). It is designed as a clear, educational, and visually rich example suitable for a university thesis.

## Overview

The **Quantum Approximate Optimization Algorithm (QAOA)** is a hybrid quantum-classical algorithm designed to find approximate solutions to combinatorial optimization problems. 

In this demo, we apply QAOA to the **Max-Cut** problem on a 4-node cycle graph. The goal of Max-Cut is to partition the nodes of a graph into two distinct sets such that the number of edges connecting nodes in different sets (the "cut" edges) is maximized.

### Key Features
- **PennyLane Integration:** Uses `pennylane` for building and simulating the quantum circuits and calculating gradients.
- **Support for Multi-Layer QAOA ($p \ge 1$):** The circuit dynamically supports any depth of QAOA layers, controlled by the $p$ parameter.
- **Classical Optimization Loop:** Implements PennyLane's `AdagradOptimizer` to classically tune the quantum parameters ($\gamma$ and $\beta$).
- **Rich CLI Dashboard:** Provides a beautiful terminal interface with live progress tracking during the optimization phase.
- **Unified Visual Dashboard:** Generates a modern, academic 1x3 Matplotlib dashboard showing:
  1. The original graph topology.
  2. The quantum probability distribution of all basis states.
  3. The final partitioned graph with visually distinct cut edges.

## Project Structure

```text
src/qaoa_maxcut/
├── __init__.py
├── circuit.py           # Defines the QAOA quantum circuit and sampling logic
├── demo.py              # Main execution script (UI, Optimization Loop, Orchestration)
├── graph_definition.py  # Defines the problem graph (e.g., 4-node cycle)
├── plotting.py          # Unified Matplotlib dashboard generation
└── qaoa_components.py   # Builds the Cost and Mixer Hamiltonians
```

## Mathematical Background

QAOA works by preparing a parameterized quantum state $|\psi(\gamma, \beta)\rangle$ through the repeated application of two unitary operators:
1.  **Cost Unitary:** $U(H_C, \gamma) = e^{-i \gamma H_C}$
2.  **Mixer Unitary:** $U(H_M, \beta) = e^{-i \beta H_M}$

Where $H_C$ is the Cost Hamiltonian representing the Max-Cut problem, and $H_M$ is the Mixer Hamiltonian (typically the sum of Pauli-X operators).

A classical optimizer then updates the vectors $\gamma$ and $\beta$ to minimize the expectation value $\langle \psi | H_C | \psi \rangle$. Once optimized, the state is measured to sample the most probable bitstrings, which correspond to the optimal cuts.

## Installation and Setup

This project uses `conda` to manage the environment.

1. Ensure you have Miniconda or Anaconda installed.
2. Create the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate qaoa-pennylane
   ```
4. Install the UI styling dependency:
   ```bash
   pip install rich
   ```

## Usage

To run the full demonstration, including the optimization loop and visual dashboard, execute the `demo.py` script from the root of the project:

```bash
PYTHONPATH=src python src/qaoa_maxcut/demo.py
```

### Expected Output
1. **Terminal:** You will see a `rich` formatted output detailing the graph, initializing parameters, and a live progress bar tracking the Adagrad optimization steps and cost reduction.
2. **Dashboard:** Upon completion, a Matplotlib window will open showing the graph, the probability histogram (expect peaks at `0101` and `1010` for a square graph), and the partitioned graph showing the cut edges in dashed dark blue.
