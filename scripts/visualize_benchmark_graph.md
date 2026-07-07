# Ispezione dei Grafi del Benchmark (`scripts/visualize_benchmark_graph.py`)

Questo file markdown descrive la logica di visualizzazione ed esplorazione interattiva dei grafi generati ed analizzati all'interno della suite di benchmark.

---

## 🧠 Modello Matematico e Interpretazione dei Grafici

Lo script consente all'utente di selezionare interattivamente uno dei grafi generati casualmente per il benchmark (salvati in formato `.gpickle` in `data/benchmarking_results/`) e visualizzarne la struttura e la soluzione esatta del taglio.

### Interpretazione del Grafico Generato

Viene mostrata una finestra di Matplotlib contenente il grafo selezionato:

* **Nodi (Vertici)**: 
  I nodi del grafo vengono suddivisi e colorati in base alla partizione ottima esatta calcolata tramite Programmazione Lineare Intera (ILP). I nodi appartenenti al **Set A (bit 0)** e al **Set B (bit 1)** assumono colori contrastanti distinti.
* **Archi (Collegamenti)**:
  * **Archi Tagliati (Cut Edges)**: Vengono disegnati con linee nere continue spesse. Rappresentano le connessioni che collegano nodi di colori diversi (la cui somma costituisce il valore ottimale del Max Cut).
  * **Archi Non Tagliati (Uncut Edges)**: Vengono disegnati con linee grigie tratteggiate sottili. Collegano nodi aventi lo stesso colore e non contribuiscono al taglio.
* **Titolo del Grafico**:
  Mostra i dettagli del grafo: il numero identificativo, il numero di vertici ($N$), il numero di archi totali ($|E|$), la densità del grafo e il valore del Max Cut teorico calcolato con PuLP.

Questo strumento di debug visivo è fondamentale per verificare graficamente che la partizione trovata separi effettivamente il maggior numero di archi possibile e per analizzare la topologia di grafi difficili da risolvere.
