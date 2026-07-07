import json
import os

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

# --- Introduction Cell ---
add_md(r"""
# Studio dell'Ottimizzazione QAOA tramite Gradient Descent (Discesa del Gradiente)
Questo notebook è dedicato all'esplorazione teorica e prima del funzionamento dell'algoritmo di **Gradient Descent (GD)** applicato alla minimizzazione della funzione di costo del **Quantum Approximate Optimization Algorithm (QAOA)** per il problema del Max-Cut.

---

## 🧠 Fondamenti Teorici e Fisici

### 1. La Funzione di Costo QAOA ($p=1$)
Nel livello più elementare di QAOA ($p=1$), lo stato variazionale quantistico $|\gamma, \beta\rangle$ è preparato applicando alternativamente l'Hamiltoniana di costo $H_C$ (che codifica le connessioni del grafo) e l'Hamiltoniana di mixer $H_M$ (che introduce sovrapposizione e fluttuazioni quantistiche):

$$|\gamma, \beta\rangle = e^{-i \beta H_M} e^{-i \gamma H_C} |+\rangle^{\otimes N}$$

L'obiettivo dell'algoritmo quantistico è massimizzare il valore atteso del taglio del grafo, ovvero $\langle C(\gamma, \beta) \rangle$. 
Tuttavia, gli ottimizzatori classici sono per convenzione progettati per **minimizzare** una funzione. Definiamo quindi la **funzione di costo** $f(\gamma, \beta)$ come il valore atteso del taglio cambiato di segno:

$$f(\gamma, \beta) = -\langle C(\gamma, \beta) \rangle$$

Nel caso ideale (es. grafo bipartito perfetto come un ciclo a 4 nodi), il valore massimo del taglio è $4.0$, perciò il minimo globale a cui tende la funzione di costo è $-4.0$.

### 2. Discesa del Gradiente (Gradient Descent)
L'ottimizzazione classica a discesa del gradiente aggiorna in modo iterativo i parametri $\gamma$ e $\beta$ muovendosi nella direzione in cui la pendenza (la derivata) del costo decresce più rapidamente.
La regola di aggiornamento è definita come:

$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$$

Dove:
* $\theta = [\beta, \gamma]^T$ è il vettore dei parametri.
* $\eta$ è il **learning rate** (tasso di apprendimento), che determina l'ampiezza di ciascun passo.
* $\nabla f(\theta) = \left[ \frac{\partial f}{\partial \beta}, \frac{\partial f}{\partial \gamma} \right]^T$ è il gradiente spaziale.

### 3. Stima Numerica del Gradiente
Dal momento che sui computer quantistici (o simulatori) non disponiamo di espressioni analitiche chiuse per il gradiente in contesti complessi, calcoliamo le derivate parziali numericamente usando il metodo delle **differenze finite centrali**:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + dx) - f(x - dx)}{2 \cdot dx}$$

In questo notebook, impostiamo una perturbazione $dx = 0.2$ per rendere la stima del gradiente solida e resiliente al rumore di campionamento probabilistico (shot noise).
""")

# --- Setup Cell ---
add_md(r"""
## Setup dell'Ambiente e Importazioni
Importiamo i moduli necessari del progetto e le librerie grafiche per visualizzare i risultati.
""")

add_code(r"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.primitives import Sampler

# Aggiunge src al path di sistema per caricare i moduli locali
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from src.qaoa.qaoa_runner import QAOARunner
from src.qaoa.optimizer import calculate_maxcut_value
""")

# --- Esecuzione GD Cell ---
add_md(r"""
## 1. Esecuzione del QAOA con Gradient Descent
Creiamo un grafo a ciclo di 4 nodi ($C_4$) ed eseguiamo il runner QAOA con l'ottimizzatore a discesa del gradiente (`GD`), salvando l'intera traiettoria percorsa dai parametri.
""")

add_code(r"""
# Creazione di un ciclo a 4 nodi (Max Cut teorico = 4.0)
G = nx.cycle_graph(4)

# Inizializziamo il runner quantistico a layer p=1
runner = QAOARunner(G, p_value=1)

# Eseguiamo il ciclo variazionale usando l'ottimizzatore a Gradient Descent
print("Avvio ottimizzazione via Gradient Descent...")
results = runner.run(
    max_optimization_iterations=25,
    optimizer_method='GD',
    tol=1e-4
)

# Estraiamo i dettagli dei risultati e la traiettoria dei parametri
opt_params = results['optimal_params']
trajectory_params = np.array(results['metrics']['trajectory_params'])

# Mappiamo i parametri nei loro intervalli periodici [0, 2*pi] per coerenza visiva
trajectory_params[:, 0] = np.mod(trajectory_params[:, 0], 2 * np.pi) # beta
trajectory_params[:, 1] = np.mod(trajectory_params[:, 1], 2 * np.pi) # gamma

print("\nOttimizzazione terminata!")
print(f"Iterazioni effettuate: {results['metrics']['optimization_iterations']}")
print(f"Parametri ottimi trovati: beta = {opt_params[0]:.4f}, gamma = {opt_params[1]:.4f}")
print(f"Valore atteso ottimo del taglio: {results['qaoa_expected_cut_value']:.4f}")
""")

# --- Visualizzazione Traiettoria 2D ---
add_md(r"""
## 2. Visualizzazione 2D della Convergenza
Tracciamo l'evoluzione dei parametri $\gamma$ (angolo del costo) e $\beta$ (angolo del mixer) passo dopo passo, associandoli al valore del costo decrescente $-\langle C \rangle$.
""")

add_code(r"""
# Utilizziamo il sampler interno al runner per coerenza statistica
sampler = runner.sampler

# Valutiamo il costo esatto lungo ciascun punto della traiettoria principale
cost_history = []
num_qubits = G.number_of_nodes()

for params in trajectory_params:
    param_dict = {}
    for param in runner.ansatz_circuit.parameters:
        if 'beta' in param.name:
            param_dict[param] = params[0]
        elif 'gamma' in param.name:
            param_dict[param] = params[1]
            
    bound_circuit = runner.ansatz_circuit.assign_parameters(param_dict)
    measured_circuit = bound_circuit.measure_all(inplace=False)
    job = sampler.run(measured_circuit, shots=1024)
    dist = job.result().quasi_dists[0]
    
    exp_val = 0.0
    for state_int, prob in dist.items():
        bitstring = format(state_int, f'0{num_qubits}b')
        exp_val += prob * calculate_maxcut_value(G, bitstring)
    cost_history.append(-exp_val)

cost_history = np.array(cost_history)
iterations = np.arange(len(cost_history))
betas = trajectory_params[:, 0]
gammas = trajectory_params[:, 1]

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Iterazioni (Step di Ottimizzazione)', fontsize=12)
ax1.set_ylabel('Angoli Variazionali (Radianti)', color=color, fontsize=12)
line1 = ax1.plot(iterations, gammas, color='royalblue', label=r'Gamma ($\gamma$)', linewidth=2.5)
line2 = ax1.plot(iterations, betas, color='deepskyblue', linestyle='--', label=r'Beta ($\beta$)', linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Costo -<C>', color=color, fontsize=12)
line3 = ax2.plot(iterations, cost_history, color='crimson', marker='o', label='Costo', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# Legenda unificata
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, shadow=True)

plt.title('Evoluzione dei Parametri e del Costo durante il Gradient Descent', fontsize=14, fontweight='bold', pad=15)
plt.show()
""")

# --- Calcolo Landscape ---
add_md(r"""
## 3. Calcolo del Panorama di Costo (3D Surface & Contour)
Per poter sovrapporre la traiettoria dell'ottimizzatore su un grafico continuo del panorama quantistico, calcoliamo i valori della funzione di costo $-\langle C(\gamma, \beta) \rangle$ su una griglia di valori di $\gamma$ e $\beta$ distribuiti tra $0$ e $2\pi$.
""")

add_code(r"""
# Calcolo della griglia (22x22 = 484 punti)
steps = 22
gamma_vals = np.linspace(0, 2 * np.pi, steps)
beta_vals = np.linspace(0, 2 * np.pi, steps)
gamma_grid, beta_grid = np.meshgrid(gamma_vals, beta_vals)
cost_grid = np.zeros_like(gamma_grid)

print("Calcolo in corso del panorama quantistico (484 valutazioni)...")
for i in range(steps):
    for j in range(steps):
        g = gamma_grid[i, j]
        b = beta_grid[i, j]
        
        param_dict = {}
        for param in runner.ansatz_circuit.parameters:
            if 'gamma' in param.name:
                param_dict[param] = g
            elif 'beta' in param.name:
                param_dict[param] = b
                
        bound_circuit = runner.ansatz_circuit.assign_parameters(param_dict)
        measured_circuit = bound_circuit.measure_all(inplace=False)
        job = sampler.run(measured_circuit, shots=1024)
        dist = job.result().quasi_dists[0]
        
        exp_val = 0.0
        for state_int, prob in dist.items():
            bitstring = format(state_int, f'0{num_qubits}b')
            exp_val += prob * calculate_maxcut_value(G, bitstring)
            
        # Assegniamo il costo negativo
        cost_grid[i, j] = -exp_val

print("Panorama quantistico calcolato con successo!")
""")

# --- Visualizzazione Traiettoria Contour ---
add_md(r"""
## 4. Visualizzazione Bidimensionale a Contorni (Contour Plot)
Un modo chiaro per vedere come l'ottimizzatore scende verso le valli di minimo è tracciare un grafico a contorni colorati (Contour Plot) in 2D e sovrapporre ad esso la traiettoria percorsa dal Gradient Descent.
""")

add_code(r"""
plt.figure(figsize=(10, 8))

# Disegna la mappa di costo riempita
cp = plt.contourf(gamma_grid, beta_grid, cost_grid, levels=30, cmap='viridis')
cbar = plt.colorbar(cp)
cbar.set_label('Costo -<C>', fontsize=12)

# Sovrappone la traiettoria GD
plt.plot(gammas, betas, color='cyan', linewidth=3, label='Traiettoria GD')
plt.scatter(gammas[0], betas[0], color='lime', edgecolor='black', s=150, zorder=5, label='Punto di Inizio')
plt.scatter(gammas[-1], betas[-1], color='red', edgecolor='black', marker='*', s=250, zorder=5, label='Ottimo (Minimo)')

plt.xlabel(r'Gamma ($\gamma$)', fontsize=12)
plt.ylabel(r'Beta ($\beta$)', fontsize=12)
plt.title('Traiettoria del Gradient Descent sovrapposta alla mappa di costo 2D', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='upper right', shadow=True)
plt.grid(True, linestyle=':', alpha=0.5)

plt.show()
""")

# --- Visualizzazione Traiettoria 3D ---
add_md(r"""
## 5. Panorama 3D del Costo e Traiettoria in 3D
Utilizziamo gli strumenti tridimensionali di Matplotlib (`mplot3d`) per renderizzare la superficie del costo e mostrare come la traiettoria scenda lungo i fianchi del panorama.
""")

add_code(r"""
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Disegna la superficie 3D del costo
surf = ax.plot_surface(gamma_grid, beta_grid, cost_grid, cmap='viridis', alpha=0.75, edgecolor='none', zorder=1)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Costo -<C>')

# Disegna la linea della traiettoria nello spazio 3D
ax.plot(gammas, betas, cost_history, color='cyan', linewidth=4, zorder=10, label='Percorso Ottimizzazione')

# Marcatori di inizio e fine
ax.scatter(gammas[0], betas[0], cost_history[0], color='lime', edgecolor='black', s=120, zorder=15, label='Inizio')
ax.scatter(gammas[-1], betas[-1], cost_history[-1], color='red', edgecolor='black', marker='*', s=200, zorder=15, label='Ottimo')

# Configurazione assi e vista
ax.set_xlabel(r'$\gamma$ (Costo)', fontsize=12)
ax.set_ylabel(r'$\beta$ (Mixer)', fontsize=12)
ax.set_zlabel(r'Costo $-\langle C \rangle$', fontsize=12)
ax.set_title('Panorama 3D del Costo e Traiettoria GD (Il percorso scende verso il minimo)', fontsize=14, fontweight='bold', pad=15)
ax.view_init(elev=35, azim=-120)
ax.legend()

plt.show()
""")

# --- Discrepancy explanation cell ---
add_md(r"""
## 🔍 Nota sulla Discrepanza dei Punti Rispetto alla Superficie Visiva

Nel grafico 3D, potresti notare che la linea ciano o la stella rossa dell'ottimo si collocano leggermente al di sotto o fluttuano in modo non perfettamente allineato rispetto alla superficie colorata disegnata. Questo fenomeno non è un errore, ma è dovuto a precisi motivi fisici e algoritmici:

1. **Risoluzione della Griglia (Discretizzazione)**: La superficie viene renderizzata su una griglia di soli $22 \times 22$ campionamenti per mantenere la visualizzazione fluida. Di conseguenza, le valli e i canyon ripidi del costo quantistico vengono "smussati", facendo apparire il fondo valle visivo meno profondo (es. fermandosi a $-2.7$) di quanto non sia in realtà.
2. **Ottimizzazione nel Continuo**: L'ottimizzatore a Gradient Descent calcola i gradienti analizzando incrementi infinitesimali nello spazio continuo $\mathbb{R}^2$ e non è limitato alla griglia discreta del grafico. Riesce quindi a infilarsi nella punta esatta e più profonda del minimo quantistico (trovando valori vicini a $-3.05$).
3. **Fluttuazioni Statistiche (Shot Noise)**: La stima quantistica dell'energia risente dell'incertezza dovuta a un numero finito di misure (1024 shots). Questo introduce piccole fluttuazioni casuali tra una misura e l'altra che accentuano geometricamente questo divario visivo locale.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "qaoa-pennylane",
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
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "Studio_Gradient_Descent.ipynb"))
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook creato con successo in: {notebook_path}")
