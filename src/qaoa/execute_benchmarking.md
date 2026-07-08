# QAOA Benchmarking Pipeline (`src/qaoa/execute_benchmarking.py`)

Questo file markdown descrive i fondamenti teorici, il ruolo del solutore classico e la metrica di valutazione del framework di benchmarking di QAOA per Max-Cut.

---

## 🧠 Modello Matematico e Fisico

Il benchmarking permette di studiare sistematicamente le prestazioni del QAOA su un dataset di grafi randomizzati con numero di nodi $N$ e densità di archi variabili, confrontando la soluzione quantistica approssimata con la soluzione classica esatta.

### Solutore Esatto Classico (Integer Linear Program - ILP)
Per grafi di grandi dimensioni, la ricerca esaustiva (brute-force) delle partizioni ottimali diventa computazionalmente intrattabile ($O(2^N)$). Il benchmark utilizza quindi la **Programmazione Lineare Intera (ILP)** formulando il problema Max-Cut come:

$$\max \sum_{(u, v) \in E} y_{uv}$$

Sotto i vincoli:
* $y_{uv} \le x_u + x_v$ per ogni $(u,v) \in E$
* $y_{uv} \le 2 - (x_u + x_v)$ per ogni $(u,v) \in E$
* $x_i \in \{0, 1\}$ (assegnamento di partizione del nodo $i$)
* $y_{uv} \in \{0, 1\}$ (indicatore che l'arco $(u,v)$ è tagliato)

Questo modello viene risolto velocemente all'interno di `src/data/exact_maxcut_solver.py` tramite la libreria `PuLP`.

### Approximation Ratio (Metrica di Performance)
La metrica fondamentale per misurare l'efficacia dell'algoritmo quantistico rispetto a quello classico è l'**Approximation Ratio (Rapporto di Approssimazione)** $\alpha$, definito come:

$$\alpha = \frac{\langle C(\gamma_{opt}, \beta_{opt}) \rangle}{C_{max}}$$

Dove:
* $\langle C(\gamma_{opt}, \beta_{opt}) \rangle$ è il valore atteso dell'operatore di taglio misurato sul processore quantistico usando i parametri variazionali ottimali trovati durante il ciclo di ottimizzazione classica.
* $C_{max}$ è la dimensione del taglio ottimo esatto calcolato dal solutore classico ILP.

Un valore di $\alpha \approx 1$ indica che QAOA ha trovato quasi perfettamente la soluzione ottimale o un'ottima approssimazione di essa.

---

## 📊 Output del Benchmark

Lo script esegue il QAOA su diversi grafi salvando un file JSON riassuntivo in `data/benchmarking_results/qaoa_benchmarking_summary.json`. I dati esportati includono:
* Metadati del grafo (nodi, archi, densità).
* Configurazione QAOA (layer $p$, tipo di mixer, encoding).
* Soluzione esatta di riferimento (dimensione del Max-Cut).
* Soluzione quantistica (taglio atteso finale, miglior bitstring misurata).
* Metriche di convergenza (storia del costo, tempo impiegato, numero di valutazioni).
* Approximation Ratio calcolato.

---

## 🛠️ Utilizzo da Riga di Comando

Lo script di benchmarking supporta diverse opzioni per agevolare i test iterativi e il confronto tra algoritmi, richiamabili direttamente da riga di comando:

* `--setup-only`: Esegue esclusivamente la fase di generazione del dataset e risoluzione tramite ILP, salvando i risultati esatti senza far partire il processo quantistico QAOA.
* `--qaoa-only`: Salta la generazione dei grafi e carica i dati preesistenti, passando direttamente all'esecuzione delle primitive Qiskit. Ottimo se i grafi di test sono già stati pre-calcolati.
* `--optimizers [ALG1] [ALG2] ...`: Sovrascrive la lista di algoritmi classici utilizzati. Si possono inserire uno o più algoritmi (es. `COBYLA`, `GD`, `SLSQP`) che verranno eseguiti in sequenza per agevolarne il confronto.

**Esempio di utilizzo:**
```bash
python -m src.qaoa.execute_benchmarking --qaoa-only --optimizers COBYLA GD
```
