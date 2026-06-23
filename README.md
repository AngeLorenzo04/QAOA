# QAOA per Max-Cut e Max-k-Cut

Questo progetto implementa l'algoritmo **Quantum Approximate Optimization Algorithm (QAOA)** per risolvere problemi di ottimizzazione combinatoria su grafi, nello specifico **Max-Cut** e **Max-k-Cut**.

L'implementazione utilizza **PennyLane**, una libreria cross-platform per il calcolo quantistico differenziabile, integrata con **NetworkX** per la gestione dei grafi.

## 🚀 Novità del Refactoring
Il progetto è stato recentemente refactorizzato per migliorare la modularità e ridurre la duplicazione del codice:
- **Core Centralizzato**: Tutta la logica di generazione circuiti e visualizzazione è ora in `src/common`.
- **Dashboard Unificata**: Un unico strumento di plotting per entrambi i problemi.
- **Interfaccia CLI Migliorata**: Script interattivi più robusti basati sulla libreria `rich`.

## 📂 Struttura del Progetto

Il codice è stato esteso introducendo pipeline di **Benchmarking** e passando all'utilizzo di **Qiskit** per l'infrastruttura quantistica più scalabile:

```text
src/
├── common/            # Utility condivise usate dalle demo classiche
│   ├── graphs.py      # Generatore di grafi (Ciclo, Completo, Casuale, Petersen, etc.)
│   ├── plotting.py    # Dashboard 1x3 unificata per la visualizzazione grafica
│   └── qaoa.py        # Factory base per i circuiti quantistici
├── data/              # Gestione dataset e soluzioni matematiche esatte
│   ├── exact_maxcut_solver.py    # Risolve il MaxCut in modo esatto: approccio Brute-Force (grafi piccoli) e ILP via PuLP (grafi grandi)
│   └── graph_dataset_generator.py # Genera dataset di grafi randomizzati (N nodi, D densità), memorizzando i .gpickle e i relativi metadati
├── max_cut/           # Modulo Max-Cut (2 partizioni)
│   ├── circuit.py     # Definizione del QNode per Max-Cut
│   ├── components.py  # Costruzione Hamiltoniane base (H_cost, H_mixer)
│   ├── execute_benchmarking.py # Script centrale di QAOA Benchmarking: orchestra generazione, calcolo ILP esatto e run quantistico, esportando il summary finale in JSON
│   └── main.py        # Demo interattiva classica per Max-Cut
├── max_k_cut/         # Modulo Max-k-Cut (k partizioni)
│   ├── circuit.py     # Definizione del QNode (n*k qubit, one-hot encoding)
│   ├── components.py  # Hamiltoniane modificate con metodo di penalità
│   └── main.py        # Demo interattiva classica per Max-k-Cut
├── qaoa/              # Modulo avanzato di esecuzione QAOA basato su Qiskit Primitives
│   ├── ansatz.py      # Crea il QAOA Ansatz parametrizzato alternando layer di Cost Hamiltonian e Mixer Hamiltonian
│   ├── encoding.py    # Strategie per la rappresentazione e codifica quantistica
│   ├── optimizer.py   # Implementa il ciclo ibrido di ottimizzazione: minimizza il valore atteso quantistico usando l'ottimizzatore classico COBYLA e Qiskit Sampler
│   └── qaoa_runner.py # Esegue una singola configurazione QAOA unendo Ansatz, Ottimizzatore ed esecuzione su un grafo specifico
└── visualization/     # Strumenti di visualizzazione avanzati
    └── plotter.py     # Generazione di grafici statistici e plot comparativi per i risultati JSON estratti dal benchmarking
```

## 🛠️ Requisiti

Assicurati di avere installato le dipendenze:
```bash
pip install -r requirements.txt
```

Librerie principali: `pennylane`, `qiskit`, `pulp`, `networkx`, `matplotlib`, `rich`, `scipy`, `tqdm`.

## 💻 Come Eseguire

Tutti i comandi devono essere eseguiti dalla root del progetto.

### Demo Max-Cut
Risolve il problema del taglio massimo (2 partizioni) su grafi selezionabili dall'utente.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/max_cut/main.py
```

### Demo Max-k-Cut
Risolve il problema del k-taglio massimo utilizzando un metodo di penalità per i vincoli.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/max_k_cut/main.py
```

### Benchmarking QAOA (Nuovo)
Esegue una massiccia simulazione automatizzata per calcolare la precisione dell'algoritmo QAOA su molteplici combinazioni di grafi (dimensione e densità). Sfrutta la libreria `pulp` per trovare le soluzioni matematicamente perfette e calcolare l'Approximation Ratio finale in JSON.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m src.max_cut.execute_benchmarking
```

### Visualizzazione dei Grafi di Benchmark (Nuovo)
Consente all'utente di scegliere in modo interattivo quale grafo usato per il benchmark visualizzare, evidenziando opzionalmente il taglio massimo esatto (Max Cut) precalcolato (o calcolato tramite ILP).
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python visualize_benchmark_graph.py
```

## 🧠 Descrizione Algoritmi

### Max-Cut
L'obiettivo è partizionare i nodi di un grafo in due set tali che il numero di archi che collegano i due set sia massimizzato. QAOA mappa questo problema su un sistema di qubit dove ogni qubit rappresenta un nodo.

### Max-k-Cut
Estensione a *k* partizioni. Utilizziamo un **one-hot encoding**: ogni nodo è rappresentato da *k* qubit. Un termine di penalità nell'Hamiltoniana di costo assicura il vincolo che ogni nodo sia assegnato a esattamente un colore:
$$H_{penalty} = \alpha \sum_i (\sum_s n_{i,s} - 1)^2$$

## 📊 Visualizzazione
Al termine di ogni esecuzione, verrà generata una **Dashboard di Analisi** contenente:
1. **Grafo del Problema**: Visualizzazione della struttura originale.
2. **Distribuzione di Probabilità**: Gli stati misurati dal computer quantistico (vengono evidenziati i più probabili).
3. **Soluzione Ottimale**: Il grafo colorato secondo la partizione migliore trovata.

---
*Progetto sviluppato nell'ambito della tesi di laurea in Informatica.*

## 🗺️ Mappa del Codice e Flusso Logico

### Grafico delle Dipendenze Modulari

Questo grafico illustra le dipendenze interne tra i vari moduli del progetto. Le frecce indicano che un modulo importa o utilizza funzionalità da un altro.

```mermaid
graph TD
    subgraph Max-k-Cut
        mkc_main["src/max_k_cut/main.py"] --> mkc_circuit("src/max_k_cut/circuit.py")
        mkc_main --> mkc_components("src/max_k_cut/components.py")
        mkc_circuit --> common_qaoa("src/common/qaoa.py")
    end

    subgraph Max-Cut
        mc_main["src/max_cut/main.py"] --> mc_circuit("src/max_cut/circuit.py")
        mc_main --> mc_components("src/max_cut/components.py")
        mc_circuit --> common_qaoa
    end

    subgraph Common
        common_qaoa
        common_graphs("src/common/graphs.py")
        common_plotting("src/common/plotting.py")
    end

    mkc_main --> common_graphs
    mkc_main --> common_plotting

    mc_main --> common_graphs
    mc_main --> common_plotting
```

### Flusso Logico dell'Applicazione

Questo diagramma rappresenta il flusso di esecuzione tipico degli script `main.py` per Max-Cut e Max-k-Cut, dopo il refactoring che ha snellito la logica.

```mermaid
flowchart TD
    start("Start (main.py)")
    start --> getUserInput("1. Get User Input (Graph & Params)")
    getUserInput --> setupQAOA("2. Setup QAOA Components (Hamiltonians, Circuits)")
    setupQAOA --> optimizeQAOA("3. Optimize QAOA Parameters")
    optimizeQAOA --> displayResults("4. Display Results (Optimal Bitstring, Plotting)")
    displayResults --> manualInspection("5. Manual Solution Inspection (Loop)")
    manualInspection --> end("End")
```

