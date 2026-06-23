import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_md("""
# Risoluzione del Problema Max-Cut tramite Algoritmi Quantistici Ibridi (QAOA)
**Presentazione del progetto**

Questo notebook illustra passo passo l'infrastruttura sviluppata per calcolare e valutare il partizionamento ottimale di grafi tramite il **Quantum Approximate Optimization Algorithm (QAOA)**, sfruttando la libreria Qiskit.
""")

add_code("""
# Setup dell'ambiente e importazione moduli
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

# Assicuriamoci che la cartella src sia nel path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from src.data.exact_maxcut_solver import find_exact_maxcut
from src.qaoa.ansatz import get_cost_hamiltonian, get_mixer_hamiltonian, get_qaoa_ansatz
from src.qaoa.qaoa_runner import QAOARunner
from src.visualization.plotter import plot_graph_with_cut, plot_optimizer_convergence, plot_probability_distribution
""")

add_md("""
## 1. Definizione del Problema (Max-Cut)
Il problema del Max-Cut consiste nel dividere i nodi di un grafo in due insiemi in modo da massimizzare il numero di archi "tagliati" (cioè che collegano un nodo del primo insieme a uno del secondo).

Creiamo un grafo di esempio a 5 nodi e calcoliamo matematicamente la soluzione perfetta (Ground Truth) usando un approccio brute-force / ILP.

> **Reference al codice:** La risoluzione esatta matematica è gestita dal modulo `src/data/exact_maxcut_solver.py`, che usa la libreria `pulp` per modellare il problema in ILP.
""")

add_code("""
# Creazione di un grafo (ciclo con un arco extra)
G = nx.cycle_graph(5)
G.add_edge(0, 2)

# Risoluzione esatta matematica
exact_result = find_exact_maxcut(G)
exact_value = exact_result['max_cut_value']
optimal_partition = exact_result['max_cut_partitions'][0]

print(f"Max-Cut Esatto: {exact_value}")
print(f"Partizione Ottimale: {optimal_partition}")

# Visualizzazione del grafo tagliato
plot_graph_with_cut(G, optimal_partition, "Soluzione Esatta Max-Cut")
""")

add_md("""
## 2. Formulazione Quantistica (Costruiamo l'Ansatz)
Per risolvere il problema quantisticamente, mappiamo ogni nodo su un Qubit (Binary Encoding). 
Costruiamo poi un **Ansatz parametrizzato**, che funge da *template* in cui le porte logiche non sono fisse, ma variano in base ad angoli liberi ($\gamma$ e $\beta$). L'Ansatz alterna due operatori per un numero di volte pari alla profondità $p$:

1.  **Cost Hamiltonian ($H_C$)**: Modella esattamente la struttura del nostro grafo. Per ogni arco che collega due nodi $i$ e $j$, applichiamo una porta di interazione $R_{ZZ}$ (ovvero $Z_i Z_j$) guidata dall'angolo $\gamma$. Questa operazione agisce sulle fasi quantistiche per "premiare" gli stati in cui i qubit connessi si trovano in partizioni diverse.
    $$H_C = \sum_{(i,j) \in E} \\frac{I - Z_i Z_j}{2}$$
2.  **Mixer Hamiltonian ($H_M$)**: Subito dopo, applichiamo il mixer, ovvero una rotazione singola $R_X$ su tutti i qubit, guidata dall'angolo $\beta$. Il mixer crea "interferenza quantistica": spinge il sistema a ruotare uscendo dallo stato attuale per esplorare nuove e inesplorate partizioni del grafo.
    $$H_M = \sum_{i \in V} X_i$$

> **Reference al codice:** La generazione algebrica degli operatori di Pauli ($Z_i Z_j$ e $X_i$) tramite `SparsePauliOp` e la costruzione fisica del circuito quantistico Qiskit con i `ParameterVector` liberi avvengono nel file `src/qaoa/ansatz.py`.
""")

add_code("""
# Profondità del circuito (layer p)
p = 1

# Estraiamo gli operatori Hamiltoniani
h_c = get_cost_hamiltonian(G)
h_m = get_mixer_hamiltonian(G)

print("Cost Hamiltonian (H_C):")
print(h_c)

print("\\nMixer Hamiltonian (H_M):")
print(h_m)

# Costruiamo il circuito quantistico
ansatz_circuit = get_qaoa_ansatz(G, p)
ansatz_circuit.draw(output='mpl')
""")

add_md("""
## 3. Ottimizzazione Variazionale (Loop Ibrido)
Ora utilizziamo l'ottimizzatore classico **COBYLA** per variare iterativamente gli angoli gamma e beta dell'Ansatz. 
L'obiettivo di COBYLA è quello di minimizzare l'Energia quantistica attesa del sistema (che corrisponde esattamente a massimizzare il valore del taglio!).

> **Reference al codice:** Il ciclo di ottimizzazione ibrido quantistico-classico è implementato in `src/qaoa/optimizer.py`, mentre l'orchestrazione di alto livello dell'esperimento QAOA avviene tramite la classe `QAOARunner` nel file `src/qaoa/qaoa_runner.py`.
""")

add_code("""
# Inizializziamo il nostro Runner automatizzato
runner = QAOARunner(graph=G, p_value=p)

# Eseguiamo il QAOA (Ottimizzazione + Campionamento)
print("Esecuzione ottimizzazione QAOA in corso...")
qaoa_results = runner.run(max_optimization_iterations=80, shots=1024)

print(f"\\nOttimizzazione completata in {qaoa_results['metrics']['optimization_iterations']} iterazioni!")
print(f"Parametri ottimi trovati: {qaoa_results['optimal_params']}")
print(f"Valore Atteso del Taglio: {qaoa_results['qaoa_expected_cut_value']:.2f}")
""")

add_md("""
## 4. Visualizzazione e Analisi delle Metriche
L'**Approximation Ratio** definisce la bontà del risultato (Valore QAOA / Valore Esatto). 
Inoltre, misurando il circuito finale per `1024 shots`, possiamo osservare la **Distribuzione di Probabilità**: gli stati ottimali (il taglio corretto) dovrebbero avere i picchi più alti.

> **Reference al codice:** Tutte le funzioni di plotting e generazione delle dashboard visive (come l'istogramma, il grafo colorato e la convergenza) si trovano nel modulo centralizzato `src/visualization/plotter.py`.
""")

add_code("""
approx_ratio = qaoa_results['best_measured_cut_value'] / exact_value
print(f"Approximation Ratio: {approx_ratio:.4f} (1.0 = Perfezione)")

# Plot della convergenza dell'energia
plot_optimizer_convergence(
    qaoa_results['metrics']['optimization_history'], 
    "Discesa dell'Ottimizzatore (COBYLA)"
)

# Plot dell'istogramma delle probabilità quantistiche misurate
optimal_bitstring = "".join(str(bit) for bit in optimal_partition)
plot_probability_distribution(
    qaoa_results['quasi_distribution'], 
    "Distribuzione Probabilità Stati Quantistici",
    optimal_bitstring=optimal_bitstring,
    top_n=15
)
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "quantum_env",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Presentazione_QAOA.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook creato con successo: Presentazione_QAOA.ipynb")
