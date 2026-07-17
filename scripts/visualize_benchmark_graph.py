#!/usr/bin/env python
import sys
import os

# Aggiunge la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
