import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_dashboard(graph: nx.Graph, probs: np.ndarray, best_bitstring: str, cost_history: list, n_wires: int) -> None:
    """
    Genera una dashboard unificata 1x3 per la dimostrazione di Max-Cut QAOA.
    
    Questa funzione crea una singola finestra Matplotlib contenente tre sottografici:
    1. La geometria del grafo del problema originale.
    2. Le probabilità di misurazione quantistica dopo l'ottimizzazione QAOA.
    3. La partizione Max-Cut finale evidenziando gli archi tagliati.
    
    Args:
        graph (nx.Graph): Il grafo del problema NetworkX originale.
        probs (np.ndarray): Array di probabilità di misurazione per ciascuno dei 2^n stati di base.
        best_bitstring (str): La stringa di bit (es. '0101') con la probabilità più alta.
        cost_history (list): Una lista che traccia il valore di aspettativa dell'Hamiltoniana di costo a ogni passo.
        n_wires (int): Il numero di qubit, che corrisponde al numero di nodi nel grafo.
    """
    # Applica uno stile moderno e pulito adatto per presentazioni accademiche e figure di tesi.
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Crea un layout di figura 1x3 (1 riga, 3 colonne)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Dashboard di Analisi QAOA Max-Cut (Soluzione: {best_bitstring})", fontsize=20, fontweight='bold', y=1.05)

    # Calcola un layout standard per i nodi del grafo da utilizzare coerentemente nei sottografici 1 e 3.
    # Un seed fisso assicura che il layout appaia uguale ogni volta che il codice viene eseguito.
    pos = nx.spring_layout(graph, seed=42)

    # ==========================================
    # --- Sottografico 1: Grafo del Problema Originale ---
    # ==========================================
    ax1 = axes[0]
    ax1.set_title("1. Grafo del Problema Originale", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')

    # ===============================================
    # --- Sottografico 2: Probabilità di Misurazione ---
    # ===============================================
    ax2 = axes[1]
    ax2.set_title("2. Probabilità di Misurazione", fontsize=16)
    
    # Genera etichette per l'asse X (tutte le possibili stringhe di bit come '00', '01', '10', '11')
    bitstrings = [format(i, f'0{n_wires}b') for i in range(2**n_wires)]
    
    # Evidenzia la barra corrispondente alla migliore soluzione trovata dall'algoritmo
    colors = ['tab:blue' if bs == best_bitstring else 'lightgray' for bs in bitstrings]
    
    ax2.bar(bitstrings, probs, color=colors)
    ax2.set_xlabel("Stringhe di Bit (Stati di Base)", fontsize=12)
    ax2.set_ylabel("Probabilità", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # =======================================
    # --- Sottografico 3: Partizione Max-Cut ---
    # =======================================
    ax3 = axes[2]
    ax3.set_title("3. Partizione Max-Cut", fontsize=16)

    # Step 3.1: Separa i nodi in due gruppi basandosi sulla soluzione della stringa di bit (0 vs 1).
    group_a = [i for i, bit in enumerate(best_bitstring) if bit == '0']
    group_b = [i for i, bit in enumerate(best_bitstring) if bit == '1']
    
    # Step 3.2: Separa gli archi in "tagliati" e "non tagliati".
    # Un arco è "tagliato" se i suoi due estremi appartengono a gruppi diversi (bit diversi).
    cut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] != best_bitstring[v]]
    uncut_edges = [(u, v) for u, v in graph.edges() if best_bitstring[u] == best_bitstring[v]]

    # Step 3.3: Disegna gli archi non tagliati (linee grigie, solide)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.5, edge_color='gray', style='solid', alpha=0.5)
    
    # Step 3.4: Disegna gli archi tagliati (prominenti, blu scuro, linee tratteggiate che rappresentano il "Confine")
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=1.5, edge_color='darkblue', style='dashed')

    # Step 3.5: Disegna i nodi, colorandoli diversamente per ogni gruppo della partizione
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_a, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, nodelist=group_b, node_color='lightcoral', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')

    # Step 3.6: Aggiunge una legenda personalizzata per spiegare i colori e gli stili degli archi
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightblue', markersize=12, label='Partizione 0', markeredgecolor='black'),
        Line2D([0], [0], color='w', marker='o', markerfacecolor='lightcoral', markersize=12, label='Partizione 1', markeredgecolor='black'),
        Line2D([0], [0], color='darkblue', lw=1.5, ls='--', label='Arco Tagliato')
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=10)
    ax3.axis('off')

    # Assicura che tutto si adatti bene prima della visualizzazione
    plt.tight_layout()
    plt.show()
