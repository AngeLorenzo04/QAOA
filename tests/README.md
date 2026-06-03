# Unit Tests per QAOA

Questa cartella contiene i test unitari per verificare la correttezza delle implementazioni di Max-Cut e Max-k-Cut.

## Struttura dei Test

- `test_max_cut.py`: Test per la costruzione delle Hamiltoniane, l'esecuzione del circuito e l'ottimizzazione di Max-Cut.
- `test_max_k_cut.py`: Test specifici per Max-k-Cut, inclusi il metodo di penalità e il supporto per diversi valori di *k*.
- `test_graphs.py`: Verifiche sulle funzioni di utility per la creazione dei grafi.

## Come Eseguire i Test

Assicurati di avere `pytest` installato nel tuo ambiente:

```bash
pip install pytest
```

Esegui i test dalla root del progetto impostando il `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest tests/
```

Oppure, se usi Windows (PowerShell):

```powershell
$env:PYTHONPATH += ";$(Get-Location)/src"
pytest tests/
```
