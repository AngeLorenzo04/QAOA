#!/usr/bin/env python
import sys
import os

# Aggiunge la cartella src al path di sistema per poter importare i moduli locali
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.visualization.plotter import plot_benchmark_graph

def main():
    try:
        plot_benchmark_graph()
    except KeyboardInterrupt:
        print("\nOperazione annullata dall'utente.")
    except Exception as e:
        print(f"\nSi è verificato un errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    main()
