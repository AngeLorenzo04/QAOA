# QAOA Demo Max-k-Cut (`src/max_k_cut/main.py`)

Questo file descrive la formulazione matematica del problema **Max-k-Cut** (taglio massimo a $k$ partizioni), la codifica quantistica con vincoli tramite Hamiltoniana di penalità e l'interpretazione dei relativi plot della dashboard.

---

## 🧠 Modello Matematico e Fisico

### Il Problema Max-k-Cut
Il problema **Max-k-Cut** estende il Max-Cut classico dividendo l'insieme dei nodi $V$ di un grafo in $k$ partizioni o "colori" disgiunti (invece di sole 2 partizioni). L'obiettivo è massimizzare il numero di archi che collegano nodi assegnati a colori diversi.

### Codifica Quantistica: One-Hot Encoding
Per mappare il problema su un registro quantistico, utilizziamo un approccio di **one-hot encoding**. 
Per ogni nodo $i \in V$, assegniamo un blocco di $k$ qubit. 
Lo stato del qubit nella posizione $s$-esima del blocco rappresenta l'assegnazione del colore al nodo $i$:
* $|0\rangle_{i,s}$: Il nodo $i$ non è colorato con il colore $s$.
* $|1\rangle_{i,s}$: Il nodo $i$ è colorato con il colore $s$.

Questo richiede in totale $N \times k$ qubit per un grafo con $N$ nodi.

### Costruzione dell'Hamiltoniana di Costo e Penalità
Dal punto di vista fisico, per garantire che lo stato finale corrisponda a una colorazione valida, dobbiamo inserire dei vincoli. Lo stato di ciascun nodo deve avere esattamente un solo qubit nello stato $|1\rangle$ (one-hot constraint).

L'Hamiltoniana totale di costo $H_C$ viene costruita come somma di due parti:
$$H_C = H_{cut} + H_{penalty}$$

1. **Hamiltoniana del Taglio ($H_{cut}$)**:
   Penalizza (aumenta l'energia di) archi che collegano nodi che hanno lo stesso colore.
   $$H_{cut} = \sum_{(u, v) \in E} \sum_{s=1}^k Z_{u,s} Z_{v,s}$$

2. **Hamiltoniana di Penalità ($H_{penalty}$)**:
   Garantisce il rispetto del vincolo matematico "un solo colore per nodo". Se un nodo viene assegnato a zero colori o a più di un colore, questo termine aumenta drasticamente l'energia complessiva dello stato, agendo come una barriera energetica che scoraggia l'ottimizzatore classico dall'esplorare quegli stati non fisici.
   $$H_{penalty} = \alpha \sum_{i \in V} \left( \sum_{s=1}^k \frac{I - Z_{i,s}}{2} - I \right)^2$$
   dove $\alpha > 0$ è un coefficiente di penalità opportunamente pesato.

---

## 📊 Interpretazione dei Grafici della Dashboard

1. **Grafo del Problema con Taglio (Problem Graph)**:
   * Rappresenta graficamente le connessioni originarie. 
   * Gli archi "tagliati" (che uniscono nodi di colori diversi) sono evidenziati visivamente per confermare l'efficacia del taglio trovato.

2. **Distribuzione di Probabilità (Probability Distribution)**:
   * Mostra le stringhe di misura a $N \times k$ bit ordinate per probabilità. 
   * A causa della presenza del termine di penalità $H_{penalty}$, le stringhe binarie non valide (che non rispettano il vincolo one-hot per ciascun nodo) avranno una probabilità teorica nulla o trascurabile, concentrando tutta la probabilità sugli stati legali che rappresentano partizioni valide.

3. **Soluzione del k-Taglio Ottimale (k-Cut Solution)**:
   * Colora i nodi del grafo usando fino a $k$ colori distinti.
   * Consente di valutare a colpo d'occhio la disposizione spaziale dei colori trovati dall'algoritmo quantistico.
