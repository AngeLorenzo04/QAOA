# QAOA per Max-Cut e Max-k-Cut

Questo progetto implementa l'algoritmo **Quantum Approximate Optimization Algorithm (QAOA)** per risolvere problemi di ottimizzazione combinatoria su grafi, nello specifico **Max-Cut** e **Max-k-Cut**.

L'implementazione utilizza **PennyLane**, una libreria cross-platform per il calcolo quantistico differenziabile, integrata con **NetworkX** per la gestione dei grafi.

### 🚀 Novità del Refactoring (Architettura Microkernel)
Il progetto è stato recentemente refactorizzato per adottare un'**Architettura Microkernel a Livelli**:
- **Microkernel Core (`src/qaoa/core/`)**: Gestisce la CLI interattiva basata su `rich`, il caricamento del dataset di grafi, e la registrazione dinamica dei plugin.
- **Plugin System (`src/qaoa/plugins/`)**: Tutte le operazioni e analisi (esecuzione QAOA, disegno di panorami energetici, ottimizzazione con Gradient Descent, benchmarking e visualizzazione statistiche) sono implementate come plugin autocontenuti e modularizzabili.
- **Piattaforma Unificata**: Un'unica interfaccia a riga di comando per selezionare qualsiasi grafo dal dataset e applicarvi qualsiasi azione (plugin) desiderata.

## 📂 Struttura del Progetto

Il codice è organizzato come segue:

```text
QAOA/
├── docs/                 # Documentazione teorica generale e analisi tesi
│   ├── benchmark_analysis.md           # Analisi qualitativa dei risultati del benchmark
│   └── qaoa_teoria_e_ottimizzazione.md # Guida al funzionamento di QAOA e Gradient Descent
├── notebooks/            # Jupyter Notebooks di presentazione e didattica
│   ├── Presentazione_QAOA.ipynb        # Notebook passo-passo per Google Colab o Jupyter
│   └── Studio_Gradient_Descent.ipynb   # Notebook focalizzato sullo studio del Gradient Descent
├── scripts/              # Script di utilità e visualizzazioni classiche (in dismissione)
│   ├── generate_gd_notebook.py         # Generatore statico del notebook di studio del GD
│   ├── generate_notebook.py            # Generatore statico del notebook di presentazione
│   ├── test_pulp.py                    # Test di integrità del solutore classico ILP
│   ├── test_sampler.py                 # Test di integrità del Sampler Qiskit
│   ├── visualize_benchmark_graph.py    # Script per ispezionare singolarmente i grafi del dataset
│   └── visualize_results.py            # Script per tracciare grafici statistici dal benchmark
├── src/                  # Codice sorgente dell'infrastruttura QAOA
│   ├── common/            # Utility condivise usate dalle demo classiche
│   │   ├── graphs.py      # Generatore di grafi (Ciclo, Completo, Casuale, Petersen, etc.)
│   │   ├── plotting.py    # Dashboard 1x3 unificata per la visualizzazione grafica
│   │   └── qaoa.py        # Factory base per i circuiti quantistici
│   ├── data/              # Gestione dataset e soluzioni matematiche esatte
│   │   ├── exact_maxcut_solver.py    # Risolve il MaxCut in modo esatto via brute-force e ILP
│   │   └── graph_dataset_generator.py # Genera dataset di grafi randomizzati (.gpickle)
│   ├── max_cut/           # Modulo Max-Cut (2 partizioni in Pennylane)
│   │   ├── circuit.py     # Definizione del QNode per Max-Cut
│   │   ├── components.py  # Costruzione Hamiltoniane base (H_cost, H_mixer)
│   │   └── main.py        # Demo interattiva classica per Max-Cut
│   ├── max_k_cut/         # Modulo Max-k-Cut (k partizioni in Pennylane)
│   │   ├── circuit.py     # Definizione del QNode (n*k qubit, one-hot encoding)
│   │   ├── components.py  # Hamiltoniane modificate con metodo di penalità
│   │   └── main.py        # Demo interattiva classica per Max-k-Cut
│   ├── qaoa/              # Modulo avanzato di esecuzione QAOA basato su Qiskit Primitives
│   │   ├── core/          # Microkernel Core per la gestione del sistema
│   │   │   ├── kernel.py           # Gestore dei plugin, CLI e caricamento dataset
│   │   │   └── plugin_interface.py # Classe base astratta per tutti i plugin
│   │   ├── plugins/       # Moduli plugin registrati nel microkernel
│   │   │   ├── benchmarking_plugin.py         # Benchmark configurabile classico ILP / quantistico QAOA
│   │   │   ├── plot_gradient_descent_plugin.py # Traiettoria GD 2D/3D sul panorama del costo
│   │   │   ├── plot_landscape_plugin.py        # Visualizza panorama di costo con label esatte e istogrammi
│   │   │   ├── run_qaoa_plugin.py              # Esecuzione singola ottimizzazione QAOA + dashboard
│   │   │   └── visualize_benchmarks_plugin.py  # Menu di visualizzazione e confronto dei benchmark
│   │   ├── ansatz.py      # Crea il QAOA Ansatz parametrizzato
│   │   ├── encoding.py    # Strategie per la rappresentazione e codifica quantistica
│   │   ├── optimizer.py   # Ciclo di ottimizzazione classica (COBYLA/GD)
│   │   ├── qaoa_runner.py # Esegue una singola configurazione QAOA
│   │   └── main.py        # Entry point unificato che lancia la piattaforma microkernel
│   └── visualization/     # Visualizzazioni originali (legacy, ora integrate come plugin)
└── tests/                # Test di unità e di integrazione (pytest)
```

## 🛠️ Requisiti

Assicurati di avere installato le dipendenze:
```bash
pip install -r requirements.txt
```

Librerie principali: `pennylane`, `qiskit`, `pulp`, `networkx`, `matplotlib`, `rich`, `scipy`, `tqdm`.

## 💻 Come Eseguire e Comandi del Progetto

Tutti i comandi devono essere eseguiti dalla directory root del progetto.

### 1. Piattaforma Microkernel QAOA Unificata (Qiskit)
Questo è il punto di ingresso principale per tutte le funzionalità di QAOA basate su Qiskit. Lanciando questo script, si avvia un'interfaccia interattiva CLI che consente di selezionare un grafo dal dataset e scegliere quale azione eseguire tramite i plugin registrati.
* **Avvio della Piattaforma**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  python src/qaoa/main.py
  ```
* **Azioni e Plugin disponibili nel Menu**:
  1. **Seleziona/Cambia grafo (`s`)**: Permette di filtrare i grafi per numero di nodi $N$, densità $D$, e ID.
  2. **Esecuzione QAOA Standard (`run_qaoa`)**: Esegue l'ottimizzazione classica (con optimizer `COBYLA`, `SLSQP` o `GD`), mostra il confronto dei tagli atteso/misurato con quello esatto classico (ILP), e mostra la dashboard grafica 1x3 al termine.
  3. **Visualizzazione Panorama 2D (`plot_landscape`)**: Calcola la mappa energetica 2D con un'analisi esatta Statevector per $N \le 12$ per garantire la simmetria dei bit. Individua e categorizza i minimi locali e globali e visualizza le partizioni associate in una legenda ad immagini.
  4. **Traiettoria Gradient Descent (`plot_gradient_descent`)**: Esegue la discesa del gradiente quantistica e proietta il percorso dei parametri sui panorami 2D e 3D del costo.
  5. **Benchmarking Suite (`benchmarking`)**: **[Plugin Globale]** Consente di calcolare i benchmark del dataset scegliendo l'ottimizzatore desiderato, e configurando se calcolare solo la risposta classica ILP, solo la risposta quantistica QAOA, o entrambe.
  6. **Analisi dei Risultati (`visualize_benchmarks`)**: **[Plugin Globale]** Permette di visualizzare le performance del benchmark selezionando quale grafico di analisi tracciare (Approx Ratio vs N, vs D, vs p, o confronto stochastico degli ottimizzatori).

### 2. Demo classiche in Pennylane (Max-Cut & Max-k-Cut)
Questi script offrono dimostrazioni classiche interattive su grafi di test predefiniti o personalizzati.

* **Demo Max-Cut**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  python src/max_cut/main.py
  ```
* **Demo Max-k-Cut**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  python src/max_k_cut/main.py
  ```
  # Esecuzione diretta (non-interattiva) specificando grafo, ottimizzatore e layers
  python src/qaoa/main.py --non-interactive -n 8 -d 0.25 -i 0 --optimizer GD -p 2 --plot
  ```


### 4. Ispezione Grafi del Benchmark
Consente di esplorare graficamente e in modo interattivo la partizione esatta calcolata tramite ILP per un qualunque grafo salvato nel dataset di benchmark.
* **Comando**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  python scripts/visualize_benchmark_graph.py
  ```
* **Documentazione Grafica**: Vedere [scripts/visualize_benchmark_graph.md](file:///home/angelo/Scrivania/UNI/Tesi/QAOA/scripts/visualize_benchmark_graph.md).

### 5. Analisi Grafica, Statistica e Infografiche Specifiche
Estrae i dati JSON generati dal benchmark. Tramite un **menu interattivo** permette di:
1. Tracciare i **grafici statistici globali** (Approximation Ratio vs N, p, densità).
2. Filtrare il dataset (per N, D, p, algoritmo, ID) per esplorare **grafici specifici** di una singola esecuzione (come la traccia di convergenza dell'ottimizzatore o l'istogramma delle probabilità).
* **Comando**:
  ```bash
  python scripts/visualize_results.py
  ```
* **Documentazione Grafica**: Vedere [scripts/visualize_results.md](file:///home/angelo/Scrivania/UNI/Tesi/QAOA/scripts/visualize_results.md).

### 6. Visualizzazione 3D Traiettoria Gradient Descent (GD)
Calcola e renderizza in una finestra GUI 3D interattiva il panorama continuo di costo $-\langle C(\gamma, \beta) \rangle$ e mostra il percorso intrapreso dall'ottimizzatore personalizzato a discesa di gradiente.
* **Comando**:
  ```bash
  python -m src.visualization.plot_gradient_descent
  ```
* **Documentazione Matematica & Discrepanza Superficie**: Vedere [src/visualization/plot_gradient_descent.md](file:///home/angelo/Scrivania/UNI/Tesi/QAOA/src/visualization/plot_gradient_descent.md).

### 6.1 Studio Panorama Energetico e Legenda Visuale dei Grafi (Benchmark)
Consente di selezionare interattivamente qualsiasi grafo dal dataset di benchmark, calcolarne il panorama 2D di densità di energia, rilevare i minimi locali e globali e visualizzarli con simboli dedicati. Le corrispondenti partizioni ottime/subottime dei grafi vengono mostrate affiancate come una "legenda ad immagini", mantenendo il grafico principale pulito e privo di testo sovrapposto.
* **Comando**:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  python src/visualization/plot_landscape_benchmark.py
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

