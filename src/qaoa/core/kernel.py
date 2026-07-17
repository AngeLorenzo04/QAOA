import os
import sys
import pickle
import networkx as nx
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from src.qaoa.core.plugin_interface import QAOACommandPlugin
from src.qaoa.qaoa_runner import QAOARunner
from src.data.graph_dataset_generator import load_graphs

class QAOAQAOMicrokernel:
    """
    The Core Microkernel system that handles plugin registration,
    graph selection, runner instantiation, and the interactive execution loop.
    """
    
    def __init__(self, graphs_dir: str = "data/generated_graphs"):
        self.graphs_dir = graphs_dir
        self.plugins: Dict[str, QAOACommandPlugin] = {}
        self.console = Console()
        self.selected_graph_info: Optional[dict] = None
        self.runner: Optional[QAOARunner] = None
        self.graphs_list: List[dict] = []
        
    def register_plugin(self, plugin: QAOACommandPlugin) -> None:
        """
        Registers a plugin to the microkernel.
        """
        if plugin.name in self.plugins:
            self.console.print(f"[yellow]Avviso: Il plugin '{plugin.name}' è già registrato. Verrà sovrascritto.[/yellow]")
        self.plugins[plugin.name] = plugin
        
    def load_dataset(self) -> None:
        """
        Loads the graph dataset from the configured directory.
        """
        if os.path.exists(self.graphs_dir):
            self.graphs_list = load_graphs(self.graphs_dir)
        else:
            self.graphs_list = []
            
    def generate_default_graphs(self) -> None:
        """
        Offers to generate a small set of default graphs if the dataset is empty.
        """
        self.console.print("[yellow]Nessun grafo trovato nella directory 'data/generated_graphs'.[/yellow]")
        gen_choice = Prompt.ask("Vuoi generare automaticamente alcuni grafi di prova?", choices=["si", "no"], default="si")
        if gen_choice.lower() in ["si", "s"]:
            self.console.print("[green]Generazione in corso...[/green]")
            from src.data.graph_dataset_generator import generate_and_save_graphs
            try:
                generate_and_save_graphs([4, 8], [0.25, 0.50], 2, self.graphs_dir)
                self.load_dataset()
            except Exception as e:
                self.console.print(f"[bold red]Errore durante la generazione dei grafi: {e}[/bold red]")
                
    def select_graph_interactively(self) -> None:
        """
        Guides the user through selecting a graph from the loaded dataset.
        """
        if not self.graphs_list:
            self.load_dataset()
            if not self.graphs_list:
                self.generate_default_graphs()
                if not self.graphs_list:
                    self.console.print("[bold red]Errore: Nessun grafo disponibile per la selezione.[/bold red]")
                    return
        
        self.console.print("\n[bold purple]=== SELEZIONE GRAFO ===[/bold purple]")
        
        # 1. Seleziona N
        available_n = sorted(list(set(g['n_vertices'] for g in self.graphs_list)))
        n_choices = [str(n) for n in available_n]
        n_selected = int(Prompt.ask("Seleziona il numero di nodi (N)", choices=n_choices, default=n_choices[0]))
        
        # Filtra per N
        graphs_filtered_n = [g for g in self.graphs_list if g['n_vertices'] == n_selected]
        
        # 2. Seleziona D
        available_d = sorted(list(set(g['density_edges'] for g in graphs_filtered_n)))
        d_choices = [f"{d:.2f}" for d in available_d]
        d_selected_str = Prompt.ask("Seleziona la densità degli archi (D)", choices=d_choices, default=d_choices[0])
        d_selected = float(d_selected_str)
        
        # Filtra per N e D
        graphs_filtered_nd = [g for g in graphs_filtered_n if abs(g['density_edges'] - d_selected) < 1e-5]
        
        # 3. Seleziona ID
        table = Table(title=f"Grafi disponibili (N={n_selected}, D={d_selected:.2f})")
        table.add_column("Indice", style="cyan", justify="right")
        table.add_column("ID Grafo", style="magenta")
        table.add_column("Seed", style="green")
        table.add_column("Nodi", style="yellow")
        table.add_column("Archi", style="blue")
        
        for idx, g in enumerate(graphs_filtered_nd):
            table.add_row(
                str(idx),
                str(g['id']),
                str(g['seed']),
                str(g['graph'].number_of_nodes()),
                str(g['graph'].number_of_edges())
            )
        self.console.print(table)
        
        idx_choices = [str(i) for i in range(len(graphs_filtered_nd))]
        idx_selected = int(Prompt.ask("Seleziona l'indice del grafo", choices=idx_choices, default="0"))
        
        self.selected_graph_info = graphs_filtered_nd[idx_selected]
        self.console.print(f"[green]Grafo caricato con successo (N={n_selected}, D={d_selected:.2f}, ID={self.selected_graph_info['id']}).[/green]")
        
        # Reset runner since graph changed
        self.runner = None
        
    def get_or_create_runner(self) -> Optional[QAOARunner]:
        """
        Returns the QAOARunner for the selected graph, instantiating it if necessary.
        """
        if not self.selected_graph_info:
            return None
            
        if self.runner is None:
            # Create default runner with p=1 (the run_qaoa plugin will ask for p dynamically)
            self.runner = QAOARunner(
                graph=self.selected_graph_info['graph'],
                p_value=1,
                mixer_type="standard",
                encoding_type="binary"
            )
            
        return self.runner

    def run(self) -> None:
        """
        Starts the main execution loop for the CLI.
        """
        self.load_dataset()
        
        while True:
            self.console.rule("[bold purple]QAOA Microkernel Platform[/bold purple]")
            
            # Show current selection state
            if self.selected_graph_info:
                g_info = (
                    f"[bold green]Grafo selezionato:[/bold green] N={self.selected_graph_info['n_vertices']}, "
                    f"D={self.selected_graph_info['density_edges']:.2f}, ID={self.selected_graph_info['id']}\n"
                )
                if self.runner:
                    g_info += f"[bold cyan]Runner QAOA attivo:[/bold cyan] p={self.runner.p_value}"
                else:
                    g_info += "[bold yellow]Runner QAOA non ancora inizializzato[/bold yellow]"
            else:
                g_info = "[bold red]Nessun grafo attualmente selezionato[/bold red]"
                
            self.console.print(Panel(g_info, title="[bold white]Stato Sistema[/bold white]", expand=False))
            
            # Print menu
            table = Table(show_header=False, box=None)
            table.add_column("Key", style="yellow bold")
            table.add_column("Desc", style="white")
            
            table.add_row("s", "Seleziona/Cambia il grafo dal dataset")
            
            for key, plugin in self.plugins.items():
                req_str = "" if plugin.requires_graph else " [plugin globale]"
                table.add_row(key, f"{plugin.description}{req_str}")
                
            table.add_row("q", "Esci dal programma")
            
            self.console.print(Panel(table, title="[bold magenta]Azioni Disponibili[/bold magenta]", expand=False))
            
            choice = Prompt.ask("Scegli un'opzione").strip().lower()
            
            if choice == 'q':
                self.console.print("[bold red]Uscita dal programma... Ciao![/bold red]")
                break
            elif choice == 's':
                self.select_graph_interactively()
            elif choice in self.plugins:
                plugin = self.plugins[choice]
                
                # Check graph requirements
                if plugin.requires_graph and not self.selected_graph_info:
                    self.console.print("[bold red]Errore: Questa azione richiede di selezionare prima un grafo (tasto 's').[/bold red]")
                    continue
                
                runner = None
                if plugin.requires_graph:
                    runner = self.get_or_create_runner()
                    if not runner:
                        continue
                
                try:
                    self.console.print(f"\n[bold green]>>> Esecuzione plugin: {plugin.name}...[/bold green]")
                    plugin.execute(self.selected_graph_info, runner, self.console)
                    self.console.print(f"[bold green]>>> Esecuzione completata.[/bold green]\n")
                except Exception as e:
                    self.console.print(f"[bold red]Errore durante l'esecuzione del plugin '{plugin.name}': {e}[/bold red]")
                    import traceback
                    self.console.print(traceback.format_exc())
            else:
                self.console.print("[bold red]Scelta non valida. Riprova.[/bold red]")
