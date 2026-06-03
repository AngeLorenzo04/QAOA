# QAOA per Max-Cut e Max-k-Cut

Questo progetto implementa l'algoritmo **Quantum Approximate Optimization Algorithm (QAOA)** per risolvere problemi di ottimizzazione combinatoria su grafi, nello specifico **Max-Cut** e **Max-k-Cut**.

L'implementazione utilizza **PennyLane**, una libreria cross-platform per il calcolo quantistico differenziabile.

## Struttura del Progetto

Il codice è organizzato in modo modulare per separare le diverse varianti del problema:

```text
src/
├── max_cut/           # Modulo dedicato al problema Max-Cut (2 partizioni)
│   ├── circuit.py     # Definizione dei circuiti quantistici (QNodes)
│   ├── components.py  # Costruzione delle Hamiltoniane di costo e mixer
│   ├── plotting.py    # Utility per la visualizzazione dei risultati
│   ├── utils.py       # Utility per la generazione dei grafi
│   └── main.py        # Script principale per eseguire la demo Max-Cut
├── max_k_cut/         # Modulo dedicato al problema Max-k-Cut (k partizioni)
│   ├── circuit.py     # Circuiti quantistici per Max-k-Cut (metodo di penalità)
│   ├── components.py  # Costruzione delle Hamiltoniane con termini di penalità
│   └── main.py        # Script principale per eseguire la demo Max-k-Cut
└── common/            # Utility condivise (grafi, plotting generico)
```

## Requisiti

Per eseguire il progetto, assicurati di avere installato le dipendenze elencate in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Le librerie principali includono:
- `pennylane`: Per la simulazione quantistica e l'ottimizzazione.
- `networkx`: Per la manipolazione e creazione dei grafi.
- `matplotlib`: Per la visualizzazione.
- `rich`: Per un'interfaccia terminale avanzata.

## Come Eseguire

### Demo Max-Cut
Per eseguire l'algoritmo Max-Cut su un grafo a ciclo di 4 nodi:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/max_cut/main.py
```

### Demo Max-k-Cut
Per eseguire l'algoritmo Max-k-Cut (con k=3) su un grafo a triangolo:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/max_k_cut/main.py
```

## Descrizione degli Algoritmi

### Max-Cut
L'obiettivo è partizionare i nodi di un grafo in due set tali che il numero di archi che collegano i due set sia massimizzato. QAOA mappa questo problema su un sistema di qubit dove ogni qubit rappresenta un nodo e lo stato (0 o 1) indica l'appartenenza a una partizione.

### Max-k-Cut
Estensione del problema precedente a *k* partizioni. Poiché PennyLane non ha un supporto nativo diretto come per Max-Cut, utilizziamo un **metodo di penalità**. Ogni nodo è rappresentato da *k* qubit (one-hot encoding); un termine di penalità nell'Hamiltoniana di costo assicura che ogni nodo sia assegnato a esattamente un colore.

---
*Progetto sviluppato nell'ambito della tesi di laurea in Informatica.*
