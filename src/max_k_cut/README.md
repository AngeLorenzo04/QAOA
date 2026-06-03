# Modulo Max-k-Cut QAOA

Questo modulo estende l'algoritmo QAOA per risolvere il problema del **k-taglio Massimo (Max-k-Cut)** utilizzando un metodo di penalità.

## Descrizione del Problema
Il problema Max-k-Cut richiede di partizionare i nodi di un grafo in $k$ set (colori) in modo da massimizzare il numero di archi che collegano nodi appartenenti a partizioni diverse.

## Implementazione (Metodo di Penalità)
Poiché non esiste una mappatura diretta qubit-partizione per $k > 2$ come in Max-Cut, utilizziamo un **one-hot encoding**. 

Ogni nodo $i$ è rappresentato da un set di $k$ qubit $\{q_{i,0}, q_{i,1}, \dots, q_{i,k-1}\}$. 
- Se il qubit $q_{i,s}$ è nello stato $|1\rangle$, il nodo $i$ è assegnato alla partizione $s$.
- Il sistema totale richiede quindi $n \times k$ qubit.

### Hamiltoniana di Costo ($H_C$)
L'Hamiltoniana di costo è composta da due termini principali:

1. **Termine Obiettivo:** Penalizza (aumenta l'energia) se due nodi adiacenti hanno lo stesso colore.
   $$H_{obj} = \sum_{(u,v) \in E} \sum_{s=0}^{k-1} n_{u,s} n_{v,s}$$
   dove $n_{i,s} = \frac{1}{2}(I - Z_{i,s})$ è l'operatore numero per il qubit $(i, s)$.

2. **Termine di Penalità (Hard Constraint):** Assicura che ogni nodo sia assegnato a **esattamente un colore**. Se un nodo ha zero o più di un colore assegnato, viene applicata una penalità pesante $\alpha$:
   $$H_{penalty} = \alpha \sum_{i=1}^{n} \left( \sum_{s=0}^{k-1} n_{i,s} - 1 \right)^2$$

### Hamiltoniana Mixer ($H_M$)
Utilizziamo un mixer di tipo $X$ su tutti gli $n \times k$ qubit. Sebbene questo mixer possa portare il sistema in stati che violano il vincolo "un solo colore per nodo", il termine di penalità nell'Hamiltoniana di costo guida l'ottimizzazione verso lo spazio delle soluzioni ammissibili.

## Struttura del Modulo
- `components.py`: Implementa la costruzione manuale dell'Hamiltoniana di costo (obiettivo + penalità) e del mixer.
- `circuit.py`: Definisce i circuiti per l'ottimizzazione e il campionamento su $n \times k$ qubit.
- `main.py`: Script interattivo per configurare $k$, scegliere il grafo ed eseguire la demo. Include la logica di **decodifica della bitstring** per estrarre la partizione dai $n \times k$ qubit.

## Considerazioni sulla Scalabilità
Il numero di qubit cresce linearmente con $k$ ($n \times k$). Per grafi grandi o valori di $k$ elevati, la simulazione classica può diventare molto onerosa in termini di memoria e tempo di calcolo.
