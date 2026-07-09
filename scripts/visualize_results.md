# Generazione dei Grafici di Benchmarking (`scripts/visualize_results.py`)

Questo file markdown spiega l'interpretazione statistica e il significato dei grafici prodotti dallo script di visualizzazione globale dei risultati dei benchmark di QAOA.

---

## 📊 Interpretazione dei Grafici Generati

Lo script analizza i dati raccolti durante la fase di benchmark quantistico e genera quattro grafici chiave all'interno di `data/benchmarking_results/`:

### 1. Approximation Ratio vs Numero di Nodi (`plot_approx_vs_N.png`)
* **Cosa mostra**: Il rapporto di approssimazione medio $\alpha = \frac{\langle C \rangle}{C_{max}}$ in funzione del numero di vertici $N$ del grafo.
* **Interpretazione Fisica**: Mostra come si comportano le prestazioni di QAOA all'aumentare delle dimensioni del problema (e quindi dello spazio di Hilbert del sistema quantistico, che cresce come $2^N$). Tipicamente, per layer $p$ bassi (es. $p=1$), l'approximation ratio tende a decrescere leggermente all'aumentare della complessità del grafo, evidenziando la necessità di circuiti più profondi per problemi di dimensioni maggiori.

### 2. Approximation Ratio vs Layer p (`plot_approx_vs_p.png`)
* **Cosa mostra**: L'andamento delle prestazioni medie dell'algoritmo al variare del numero di passi/layer variazionali $H_C$ e $H_M$ ($p$-value).
* **Interpretazione Fisica**: Il teorema adiabatico garantisce che nel limite $p \to \infty$, l'algoritmo QAOA converge deterministicamente alla soluzione esatta del taglio (approximation ratio = $1.0$). Questo grafico permette di visualizzare empiricamente questo comportamento: all'aumentare di $p$, le prestazioni medie dovrebbero salire, a fronte però di tempi di ottimizzazione più lunghi e circuiti più sensibili al rumore.

### 3. Approximation Ratio vs Densità Archi (`plot_approx_vs_density.png`)
* **Cosa mostra**: L'approximation ratio in funzione della densità degli archi del grafo ($d = \frac{2|E|}{N(N-1)}$).
* **Interpretazione Fisica**: Studia la vulnerabilità dell'algoritmo rispetto alla "connettività" del problema. I grafici molto sparsi o molto densi possono presentare complessità diverse per l'ottimizzazione classica a causa della comparsa di minimi locali nel panorama energetico.

### 4. Convergenza dell'Ottimizzatore classico (Specifico per singolo run)
* **Cosa mostra**: Il valore della funzione di costo (valore atteso negativo del taglio $-\langle C \rangle$) valutato ad ogni iterazione classica per una specifica esecuzione scelta dall'utente.
* **Interpretazione Fisica**: Visualizza la dinamica dell'ottimizzatore classico (es. COBYLA o GD). Permette di valutare se l'ottimizzatore ha raggiunto un plateau stabile (convergenza completa) o se ha terminato i passi a disposizione prima di trovare il minimo locale/globale.

### 5. Distribuzione di Probabilità (Specifico per singolo run)
* **Cosa mostra**: Gli stati di base (bitstring) misurati al termine del circuito quantistico e la loro rispettiva probabilità, con lo stato ottimale evidenziato.
* **Interpretazione Fisica**: Mostra in che modo QAOA concentra l'ampiezza di probabilità sulle soluzioni ottimali. Permette di valutare la "qualità" dello stato preparato dal circuito.

---

## 🕹️ Menu Interattivo e Filtraggio

L'esecuzione dello script `python scripts/visualize_results.py` avvia ora un menu interattivo basato su `rich` che permette due modalità di lavoro:

1. **Grafici Globali Aggregati**: Genera massivamente i grafici 1, 2 e 3 salvandoli automaticamente su disco.
2. **Grafici Specifici**: Permette di restringere il dataset di benchmark tramite una serie di prompt (scegliendo dinamicamente `N`, la densità `D`, il layer `p`, l'algoritmo di ottimizzazione e l'ID del grafo) fino a isolare una **singola esecuzione**. Da qui, è possibile visualizzare a tutto schermo la Curva di Convergenza (4) o la Distribuzione di Probabilità (5) relativa esclusivamente a quella particolare configurazione.
