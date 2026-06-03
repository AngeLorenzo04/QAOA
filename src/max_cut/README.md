# Modulo Max-Cut QAOA

Questo modulo implementa l'algoritmo **Quantum Approximate Optimization Algorithm (QAOA)** per risolvere il problema del **Taglio Massimo (Max-Cut)**.

## Descrizione del Problema
Il problema Max-Cut consiste nel partizionare i nodi di un grafo in due set ($S_0$ e $S_1$) in modo da massimizzare il numero di archi che collegano un nodo in $S_0$ a un nodo in $S_1$.

## Implementazione
In questo modulo, il problema viene mappato su un sistema quantistico di $n$ qubit, dove $n$ è il numero di nodi del grafo. Ogni qubit $i$ rappresenta un nodo e il suo stato di base $|0\rangle$ o $|1\rangle$ indica l'appartenenza a una delle due partizioni.

### Hamiltoniane
- **Hamiltoniana di Costo ($H_C$):** Codifica la funzione obiettivo. Utilizziamo l'implementazione nativa di PennyLane `qml.qaoa.maxcut(graph)`, che costruisce un'Hamiltoniana basata su operatori di Pauli-Z:
  $$H_C = \sum_{(u,v) \in E} \frac{1}{2}(I - Z_u Z_v)$$
- **Hamiltoniana Mixer ($H_M$):** Utilizziamo il mixer standard $X$ che applica rotazioni di Pauli-X su tutti i qubit per consentire transizioni tra le diverse configurazioni di taglio.

## Struttura del Modulo
- `components.py`: Contiene la logica per costruire le Hamiltoniane di costo e mixer.
- `circuit.py`: Definisce i circuiti quantistici (QNodes) per l'ottimizzazione e per il campionamento finale dei risultati.
- `main.py`: Script interattivo che guida l'utente nella scelta del grafo, nell'esecuzione dell'algoritmo e nella visualizzazione della dashboard dei risultati.

## Come Funziona
1. **Inizializzazione:** Si prepara uno stato di sovrapposizione equa di tutti i possibili tagli (porte Hadamard su tutti i qubit).
2. **Layer QAOA ($p$ volte):** Si applicano alternativamente l'evoluzione unitaria di $H_C$ (parametro $\gamma$) e di $H_M$ (parametro $\beta$).
3. **Ottimizzazione:** Un ottimizzatore classico (es. `Adagrad`) aggiorna i parametri $\gamma$ e $\beta$ per minimizzare il valore di aspettativa $\langle \psi(\gamma, \beta) | H_C | \psi(\gamma, \beta) \rangle$.
4. **Campionamento:** Si misura lo stato finale per ottenere la stringa di bit (bitstring) che rappresenta la partizione ottimale.
