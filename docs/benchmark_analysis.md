# Analisi e Interpretazione dei Risultati QAOA per il Max-Cut

Il benchmarking del QAOA è stato completato su un totale di **720 simulazioni** (raggruppate per dimensione dei grafi $N$, densità degli archi $D$ e profondità del circuito $p$). I risultati descrivono le performance dell'algoritmo nel risolvere il problema del taglio massimo.

Ecco i trend principali estratti dall'enorme mole di dati JSON prodotti.

---

## 1. Impatto della Dimensione del Grafo ($N$)
L'osservazione più chiara riguarda il modo in cui l'algoritmo si comporta all'aumentare dei nodi:

| Dimensione ($N$) | Approximation Ratio Medio |
| :---: | :---: |
| 4 | ~0.8000 |
| 8 | ~0.9860 |
| 16 | ~0.9600 |

> [!NOTE]
> Sorprendentemente, i grafi da 8 nodi ottengono i risultati migliori, rasentando la perfezione ($> 98\%$). I grafi molto piccoli ($N=4$) mostrano un valore più basso a causa dell'alta presenza di casi degeneri con densità molto basse (ad esempio grafi disconnessi o senza archi in cui il MaxCut è 0), che abbassano drasticamente la media. Nei grafi da 16 nodi, vediamo il fisiologico, ma lento, declino delle prestazioni previsto per i sistemi quantistici in scala, che si mantengono comunque eccellenti ($96\%$).

## 2. Impatto della Densità degli Archi ($D$)
La densità gioca un ruolo fondamentale per l'accuratezza del QAOA:

- **Bassa densità ($D = 10\%$ o $25\%$):** Presenta la maggiore incertezza. A $N=4, D=10\%$, il ratio scende fino al $45\%$. Questo è dovuto al fatto che l'algoritmo fatica a creare l'appropriato grado di *interferenza quantistica* quando le interazioni tra i qubit (gli archi) sono scarse.
- **Alta densità ($D = 50\%$ o $75\%$):** Registra valori perfetti o quasi perfetti ($100\%$ per $N=4$, $99.7\%$ per $N=8$, e $\sim 96\%$ per $N=16$). Un alto numero di connessioni aiuta la funzione di costo e i layer dell'ansatz a mescolare uniformemente le probabilità, riducendo l'impatto dei minimi locali.

> [!TIP]
> **Implicazione pratica:** Il QAOA, unito a un ottimizzatore classico, performa molto meglio su grafi strutturati e densi (che per l'approccio classico brute-force sarebbero invece più complessi da calcolare).

## 3. L'anomalia della Profondità del Circuito ($p$)
La teoria del QAOA suggerisce che, per $p \to \infty$, l'Approximation Ratio tende a 1. Aumentare $p$ (i layer) dovrebbe migliorare progressivamente il risultato. Tuttavia, i nostri dati mostrano una tendenza diversa:

| Profondità ($p$) | Approximation Ratio Medio (totale) |
| :---: | :---: |
| 1 | 0.9158 |
| 2 | 0.9155 |
| 3 | 0.9140 |

> [!WARNING]
> Aumentare $p$ non ha apportato alcun beneficio pratico, anzi, in media c'è un impercettibile degrado! 

### Perché succede questo?
Questa è la celebre **"sfida dell'ottimizzazione ibrida"** ed è causata dall'ottimizzatore classico scelto (`COBYLA`). Aumentando $p$:
1. Aumenta linearmente il numero di parametri ($2p$) da fornire all'ottimizzatore.
2. Il "paesaggio" dell'energia diventa frastagliato: nascono molti *Barren Plateaus* e minimi locali.
3. Ottimizzatori "gradient-free" come COBYLA faticano immensamente a destreggiarsi in paesaggi ad alta dimensionalità e si incastrano precocemente in soluzioni sub-ottimali. 

## Conclusione e Prossimi Passi

1. L'algoritmo ha **un'eccellente accuratezza per $p=1$ e $N \le 16$** e risulta molto affidabile. 
2. **Collo di Bottiglia Classico:** Non vi è convenienza nell'aumentare la profondità del circuito ($p>1$) con l'attuale pipeline. Per sbloccare la potenza di profondità maggiori serve un ottimizzatore quantistico specializzato basato sul gradiente (come l'**SPSA** per hardware rumorosi, o metodi analitici per l'Ansatz QAOA).
3. **Hardware:** Il benchmark ha confermato il noto problema della memoria ($64$ GB per simulare perfettamente N=32), rafforzando il bisogno di eseguire questi task non con simulatori esatti ma tramite tensor networks o computer quantistici reali.
