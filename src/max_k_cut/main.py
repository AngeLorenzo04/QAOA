import pennylane as qml
from pennylane import numpy as np
import networkx as nx

# Importazioni dai moduli locali
from max_k_cut.components import build_max_k_cut_hamiltonians
from max_k_cut.circuit import create_max_k_cut_circuit, create_max_k_cut_sampling_circuit

def decode_bitstring(bitstring, n, k):
    """
    Converte una stringa di bit (lunghezza n*k) in un dizionario di colori dei nodi.
    
    Args:
        bitstring (str): La stringa di bit risultante dalla misurazione.
        n (int): Numero di nodi nel grafo.
        k (int): Numero di partizioni (colori).
        
    Returns:
        dict: Mappa nodo -> colore.
    """
    colors = {}
    for i in range(n):
        node_bits = bitstring[i*k : (i+1)*k]
        # Trova quale indice è 1 (il colore)
        # Nota: Nel metodo di penalità, potremmo avere più 1 o nessuno se l'ottimizzazione è scarsa.
        # Prendiamo il primo 1 trovato.
        try:
            color = np.where(np.array(list(node_bits)) == '1')[0][0]
            colors[i] = color
        except IndexError:
            colors[i] = -1  # Nessun colore assegnato validamente
    return colors

def run_max_k_cut_demo():
    """
    Esegue una dimostrazione del problema Max-k-Cut utilizzando QAOA.
    """
    # 1. Configurazione del problema
    k = 3
    # Grafo a triangolo: ogni nodo è connesso agli altri due. 
    # Per Max-3-Cut, la soluzione ottimale assegna un colore diverso a ogni nodo.
    graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    n_nodes = len(graph.nodes)
    
    print(f"Risoluzione di Max-{k}-Cut per un grafo a triangolo (Metodo di Penalità)...")
    
    # 2. Costruzione delle Hamiltoniane
    # Usiamo un peso di penalità per forzare il vincolo di un solo colore per nodo.
    cost_h, mixer_h = build_max_k_cut_hamiltonians(graph, k, penalty_weight=2.0)
    
    # 3. Creazione dei Circuiti
    qaoa_circuit = create_max_k_cut_circuit(graph, k, cost_h, mixer_h)
    sampling_circuit = create_max_k_cut_sampling_circuit(graph, k, cost_h, mixer_h)
    
    # 4. Ottimizzazione
    n_layers = 4
    # Inizializzazione dei parametri (gamma e beta)
    params = np.array([np.random.uniform(0.01, 0.1, n_layers) for _ in range(2)], requires_grad=True)
    
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    steps = 150
    
    print(f"Costo Iniziale: {qaoa_circuit(params):.4f}")
    print("Inizio ottimizzazione...")
    for i in range(steps):
        params = optimizer.step(qaoa_circuit, params)
        if (i + 1) % 30 == 0:
            current_cost = qaoa_circuit(params)
            print(f"Passo {i+1}/{steps}, Costo: {current_cost:.4f}")

    # 5. Risultati
    probs = sampling_circuit(params)
    best_idx = np.argmax(probs)
    n_qubits = n_nodes * k
    best_bitstring = format(best_idx, f'0{n_qubits}b')
    
    colors = decode_bitstring(best_bitstring, n_nodes, k)
    print("\nMigliore assegnazione di colori trovata:")
    for node, color in colors.items():
        print(f"Nodo {node}: Colore {color}")

if __name__ == "__main__":
    run_max_k_cut_demo()
