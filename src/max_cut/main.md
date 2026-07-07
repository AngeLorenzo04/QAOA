# QAOA Demo Max-Cut (`src/max_cut/main.py`)

Questo file markdown descrive i fondamenti matematici e fisici dello script di demo interattiva per il problema **Max-Cut** e fornisce una guida all'interpretazione dei grafici generati.

---

## 🧠 Modello Matematico e Fisico

### Il Problema Max-Cut
Dato un grafo non orientato $G = (V, E)$, il problema del **Taglio Massimo (Max-Cut)** consiste nel partizionare l'insieme dei nodi $V$ in due sottoinsiemi disgiunti $A$ e $B$ tali da massimizzare il numero (o il peso) degli archi che collegano un nodo in $A$ e un nodo in $B$.

Associando a ogni nodo $i \in V$ una variabile binaria classica $s_i \in \{-1, +1\}$, dove:
* $s_i = +1$ se il nodo $i$ appartiene ad $A$
* $s_i = -1$ se il nodo $i$ appartiene ad $B$

La dimensione del taglio è data dalla funzione:
$$C(s) = \sum_{(u, v) \in E} \frac{1 - s_u s_v}{2}$$

### Mappatura Quantistica (Hamiltoniana)
In computazione quantistica, mappiamo lo spin classico $s_i$ sull'operatore di Pauli $Z_i$ agente sul qubit $i$. 
L'operatore Hamiltoniano di costo $H_C$ corrispondente alla funzione del taglio diventa:
$$H_C = \sum_{(u, v) \in E} \frac{I - Z_u Z_v}{2}$$

L'obiettivo di QAOA è trovare lo stato quantistico che massimizza il valore atteso di questa Hamiltoniana. Dal punto di vista della fisica, lo stato fondamentale (ground state) dell'Hamiltoniana di costo corrispondente al problema di minimizzazione (cioè $-H_C$) codifica la configurazione di spin a energia minima che risolve il problema.

### L'Ansatz QAOA
L'algoritmo prepara lo stato quantistico alternando layer di evoluzione temporale sotto l'Hamiltoniana di costo $H_C$ e sotto l'Hamiltoniana di mixer $H_M$:
$$|\gamma, \beta\rangle = \prod_{k=1}^p e^{-i \beta_k H_M} e^{-i \gamma_k H_C} |+\rangle^{\otimes n}$$
dove:
* $H_M = \sum_{i \in V} X_i$ è l'Hamiltoniana di mixer (che induce fluttuazioni quantistiche e transizioni tra gli stati di spin).
* $\gamma$ e $\beta$ sono parametri variazionali classici da ottimizzare.
* $|+\rangle^{\otimes n}$ è lo stato di sovrapposizione uniforme iniziale.

---

## 📊 Interpretazione dei Grafici della Dashboard

Al termine dell'esecuzione, viene mostrata una dashboard interattiva divisa in tre pannelli principali:

1. **Grafo del Problema (Problem Graph)**:
   * Mostra la struttura topologica del grafo inserito dall'utente.
   * Evidenzia in rosso (o con tratti spessi) gli archi che sono stati tagliati (ossia che collegano nodi appartenenti a set diversi) per rendere immediata la verifica visiva della partizione trovata.

2. **Distribuzione di Probabilità degli Stati (Probability Distribution)**:
   * Rappresenta la probabilità di misurare ciascuna stringa binaria (stato computazionale) nello stato finale ottimizzato $|\gamma, \beta\rangle$.
   * Le barre più alte indicano le soluzioni candidate di Max-Cut consigliate dall'algoritmo quantistico. A causa della simmetria del problema Max-Cut (invertendo tutti gli spin il taglio rimane invariato), si noteranno sempre coppie simmetriche di stringhe binarie con la stessa identica probabilità (es. `0110` e `1001`).

3. **Soluzione Ottimale (Optimal Partition)**:
   * Colora i nodi del grafo originale in due colori diversi (es. Set A e Set B) in base alla stringa binaria a massima probabilità misurata dal circuito quantistico.
