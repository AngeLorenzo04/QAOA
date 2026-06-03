import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_k_cut_dashboard(graph: nx.Graph, k: int, probs: np.ndarray, best_bitstring: str, node_colors: dict, cost_history: list) -> None:
    """
    Genera una dashboard unificata per il problema Max-k-Cut.
    
    Args:
        graph (nx.Graph): Il grafo originale.
        k (int): Numero di colori.
        probs (np.ndarray): Distribuzione di probabilità degli stati.
        best_bitstring (str): La stringa di bit con probabilità massima.
        node_colors (dict): Mappatura nodo -> indice colore (0 a k-1).
        cost_history (list): Storia del costo durante l'ottimizzazione.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    n_nodes = len(graph.nodes)
    n_qubits = len(best_bitstring)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Analisi QAOA Max-{k}-Cut (Soluzione: {best_bitstring})", fontsize=20, fontweight='bold', y=1.05)
    
    pos = nx.spring_layout(graph, seed=42)
    
    # --- 1. Grafo Originale ---
    ax1 = axes[0]
    ax1.set_title("1. Grafo del Problema", fontsize=16)
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color='#e0e0e0', 
            node_size=800, font_size=14, font_weight='bold', edgecolors='black')
    
    # --- 2. Probabilità (Top 10 stati per leggibilità) ---
    ax2 = axes[1]
    ax2.set_title(f"2. Probabilità (Top 10)", fontsize=16)
    
    # Prendiamo solo i top 10 per non affollare il grafico se n*k è grande
    top_indices = np.argsort(probs)[-10:][::-1]
    top_probs = probs[top_indices]
    top_labels = [format(i, f'0{n_qubits}b') for i in top_indices]
    
    colors = ['tab:blue' if label == best_bitstring else 'lightgray' for label in top_labels]
    ax2.bar(top_labels, top_probs, color=colors)
    ax2.set_ylabel("Probabilità")
    ax2.tick_params(axis='x', rotation=45)
    
    # --- 3. Partizione k-Cut ---
    ax3 = axes[2]
    ax3.set_title(f"3. Partizione {k}-Cut", fontsize=16)
    
    # Palette di colori per k partizioni
    cmap = plt.get_cmap('tab10')
    
    # Colori dei nodi basati sulla soluzione
    # Se un nodo non ha colore valido (-1), lo coloriamo di grigio
    draw_colors = [cmap(node_colors[i]) if node_colors[i] != -1 else 'gray' for i in range(n_nodes)]
    
    # Identifica archi "tagliati" (nodi con colori diversi)
    cut_edges = []
    uncut_edges = []
    for u, v in graph.edges():
        if node_colors[u] != node_colors[v] and node_colors[u] != -1 and node_colors[v] != -1:
            cut_edges.append((u, v))
        else:
            uncut_edges.append((u, v))
            
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=uncut_edges, width=1.0, edge_color='gray', alpha=0.3)
    nx.draw_networkx_edges(graph, pos, ax=ax3, edgelist=cut_edges, width=2.0, edge_color='black', style='dashed')
    nx.draw_networkx_nodes(graph, pos, ax=ax3, node_color=draw_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, ax=ax3, font_size=14, font_weight='bold')
    
    # Legenda colori
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor=cmap(i), markersize=10, label=f'Colore {i}')
        for i in range(k)
    ]
    legend_elements.append(Line2D([0], [0], color='black', lw=2, ls='--', label='Arco Tagliato'))
    ax3.legend(handles=legend_elements, loc='best', fontsize=8)
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
