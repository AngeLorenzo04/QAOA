from abc import ABC, abstractmethod
import networkx as nx
from src.qaoa.qaoa_runner import QAOARunner

class QAOACommandPlugin(ABC):
    """
    Abstract base class representing a command plugin in the QAOA Microkernel architecture.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The unique identifier for the plugin (e.g., 'run_qaoa', 'plot_landscape').
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        A human-readable description of what the plugin does, displayed in the CLI menus.
        """
        pass

    @property
    def requires_graph(self) -> bool:
        """
        Whether this plugin requires a pre-selected graph from the dataset to execute.
        Defaults to True.
        """
        return True

    @abstractmethod
    def execute(self, graph_info: dict, runner: QAOARunner, console) -> None:
        """
        Executes the plugin's action.

        Args:
            graph_info (dict): Metadata and reference to the selected NetworkX graph,
                               or None if requires_graph is False.
            runner (QAOARunner): An initialized QAOARunner for the selected graph,
                                 or None if requires_graph is False.
            console: A rich Console instance for output.
        """
        pass
