# Traiettoria Gradient Descent in 3D (`src/visualization/plot_gradient_descent.py`)

Questo file markdown descrive l'interpretazione matematica, fisica e grafica dello script che visualizza la convergenza di QAOA tramite un ottimizzatore personalizzato di **Gradient Descent (GD)** sul panorama 3D della funzione di costo.

---

## 🧠 Modello Matematico e Fisico del Costo 3D

Nello spazio di ricerca di QAOA a livello $p=1$, lo stato è completamente determinato da due parametri d'angolo: $\gamma$ (angolo del costo) e $\beta$ (angolo del mixer). 
Ad ogni coppia $(\gamma, \beta)$, la fisica quantistica associa un valore atteso del taglio del problema $\langle C(\gamma, \beta) \rangle$.

Poiché l'ottimizzatore classico punta a **minimizzare** la funzione di costo, definiamo la funzione di costo $f(\gamma, \beta)$ da minimizzare come il valore atteso del taglio cambiato di segno:
$$f(\gamma, \beta) = -\langle C(\gamma, \beta) \rangle$$

Il grafico 3D rappresenta questa superficie di potenziale energetico:
* Le **valli** (regioni più basse e scure, di colore viola) rappresentano le soluzioni ottimali in cui il costo è minimo (e quindi il taglio massimo del grafo è massimizzato).
* I **picchi** (regioni rialzate e gialle) rappresentano le soluzioni peggiori, con taglio vicino a 0.

Il Gradient Descent classico aggiorna i parametri muovendosi nella direzione opposta a quella del gradiente locale:
$$\vec{\theta}_{t+1} = \vec{\theta}_t - \eta \nabla f(\vec{\theta}_t)$$
dove $\vec{\theta} = (\beta, \gamma)^T$ e $\eta$ è il learning rate. Visivamente, questa dinamica equivale a far scivolare una pallina lungo i fianchi della collina fino ad adagiarsi sul fondo della valle più profonda (il minimo globale o locale).

---

## 📈 Varianti del Gradient Descent Implementate

L'ottimizzatore personalizzato (definito in [optimizer.py](file:///home/angelo/Scrivania/UNI/Tesi/QAOA/src/qaoa/optimizer.py#L141)) calcola numericamente il gradiente a ogni passo tramite **differenze finite centrali** con un intervallo $dx$ (di default pari a $0.2$). Supporta tre diverse varianti per l'aggiornamento dei parametri:

### 1. Vanilla Gradient Descent (`vanilla`)
Aggiorna i parametri muovendosi unicamente nella direzione opposta al gradiente stimato locale:
$$\vec{\theta}_{t+1} = \vec{\theta}_t - \eta \nabla f(\vec{\theta}_t)$$

### 2. Momentum Gradient Descent (`momentum`)
Introduce una componente di "inerzia" per accumulare la velocità dei passi precedenti e superare eventuali plateau o piccoli minimi locali ostici:
$$\vec{v}_{t+1} = 0.9 \cdot \vec{v}_t + \eta \nabla f(\vec{\theta}_t)$$
$$\vec{\theta}_{t+1} = \vec{\theta}_t - \vec{v}_{t+1}$$

### 3. Adam (Adaptive Moment Estimation) (`adam`) — *Metodo di Default*
Regola in modo intelligente e dinamico il passo (learning rate) per ciascun parametro. Può essere compreso intuitivamente tramite due concetti:
1. **Memoria della direzione (Inerzia)**: Come per il *Momentum*, tiene conto dei passi precedenti per non farsi deviare da improvvise fluttuazioni o rumore.
2. **Frenata assistita nei tratti instabili**: Controlla quanto le stime del gradiente stanno oscillando per ciascun parametro. Se un parametro oscilla in modo caotico, Adam rimpicciolisce il passo per evitare di superare il minimo; se il percorso è stabile e sicuro, allarga il passo per muoversi più velocemente.

**Perché è perfetto per il QAOA?** 
I computer quantistici e i simulatori stimano il costo campionando lo stato con un numero finito di misure (shots). Questo introduce un rumore di fondo (*shot noise*) che fa fluttuare i gradienti. Adam, smorzando le oscillazioni e adattando i passi, evita che l'ottimizzatore si perda a causa del rumore.

*Formulazione matematica:*
$$\vec{m}_t = \beta_1 \vec{m}_{t-1} + (1-\beta_1) \vec{g}_t, \quad \vec{v}_t = \beta_2 \vec{v}_{t-1} + (1-\beta_2) \vec{g}_t^2$$
Dopo aver corretto il bias ($\hat{m}_t$ e $\hat{v}_t$), i parametri vengono aggiornati come:
$$\vec{\theta}_{t+1} = \vec{\theta}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon_{\text{adam}}} \hat{m}_t$$

In tutte e tre le modalità, dopo ogni passo viene applicato un **wrapping periodico** tramite operatore modulo per mantenere i parametri entro i domini fisici significativi di QAOA ($\beta \in [0, \pi]$ e $\gamma \in [0, 2\pi]$).

---

## 📊 Interpretazione dei Grafici

Lo script genera due finestre grafiche distinte:

### Finestra 1: Grafo Soluzione e Curve di Convergenza 2D
* **Pannello Sinistro**: Visualizza la partizione ottima trovata dal Gradient Descent sul grafo del problema.
* **Pannello Destro**: Mostra l'evoluzione temporale (passo dopo passo) dei parametri $\gamma$ (linea blu continua) e $\beta$ (linea celeste tratteggiata) sull'asse delle ordinate sinistro. Sull'asse delle ordinate destro viene tracciato l'andamento del costo $f(\gamma, \beta) = -\langle C \rangle$ (linea rossa con marker `x`), che mostra una discesa monotona verso il minimo teorico del problema (che per un ciclo a 4 nodi è $-4.0$).

### Finestra 2: Panorama 3D del Costo e Traiettoria GD
* Rappresenta la superficie 3D di costo $-\langle C(\gamma, \beta) \rangle$ su un dominio periodico $[0, 2\pi] \times [0, 2\pi]$.
* Disegna una traiettoria tridimensionale continua di colore ciano che unisce i vari passi dell'ottimizzatore (punti colorati che passano dal verde del punto di inizio al rosso del punto ottimo finale).

---

## 🔍 Perché il punto ottimo finale non coincide perfettamente con la superficie visiva?

Durante l'esplorazione del grafico 3D, si può notare che il punto finale di ottimo (la stella rossa `Ottimo`) o i punti della traiettoria sembrano trovarsi leggermente al di sopra della superficie o "galleggiare" su di essa. Questo comportamento è perfettamente normale ed è causato da tre fattori tecnici e fisici ben precisi:

1. **Effetto della Discretizzazione della Griglia (Coarse Grid)**:
   Per garantire che la finestra 3D interattiva risponda fluidamente ai comandi del mouse (rotazione, zoom e spostamento in tempo reale) senza subire lag, la superficie 3D viene calcolata su una griglia discretizzata relativamente grossolana (es. $22 \times 22 = 484$ punti totali).
   Poiché le valli del panorama QAOA sono molto strette e ripide, questa discretizzazione finisce per **"mancare" il punto di minimo esatto** della gola, portando a una **sottostima visiva della profondità della valle** sulla superficie renderizzata (la quale mostrerà un fondo valle visivo limitato, ad esempio, a $-2.7$).

2. **Ottimizzazione nel Dominio Continuo**:
   L'ottimizzatore a Gradient Descent si sposta all'interno dello spazio dei parametri in modo continuo e non è vincolato alla griglia discreta di discretizzazione del plot. Di conseguenza, il GD riesce a scendere fino alla **punta più profonda del minimo reale continuo** (es. trovando il costo effettivo di $-3.05$), che si trova geometricamente più in basso rispetto alla superficie discretizzata visualizzata.

3. **Rumore di Campionamento Statistico (Shot Noise)**:
   Sia il calcolo della griglia che i passi del Gradient Descent stimano il valore atteso $-\langle C \rangle$ mediante un campionamento probabilistico con un numero finito di misure (1024 shot sul Qiskit `Sampler`). Questo introduce una fluttuazione statistica intrinseca (rumore). Il punto ottimo finale può aver registrato una fluttuazione favorevole verso il basso, mentre i punti vicini sulla griglia visualizzata possono aver registrato fluttuazioni medie o positive, accentuando il divario visivo temporaneo.
