import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.qaoa.core.kernel import QAOAQAOMicrokernel
from src.qaoa.plugins.run_qaoa_plugin import RunQAOAPlugin
from src.qaoa.plugins.plot_landscape_plugin import PlotLandscapePlugin
from src.qaoa.plugins.plot_gradient_descent_plugin import PlotGradientDescentPlugin
from src.qaoa.plugins.benchmarking_plugin import BenchmarkingPlugin
from src.qaoa.plugins.visualize_benchmarks_plugin import VisualizeBenchmarksPlugin

def main():
    # Instantiate the Microkernel
    kernel = QAOAQAOMicrokernel(graphs_dir="data/generated_graphs")
    
    # Register all plugins
    kernel.register_plugin(RunQAOAPlugin())
    kernel.register_plugin(PlotLandscapePlugin())
    kernel.register_plugin(BenchmarkingPlugin())
    kernel.register_plugin(VisualizeBenchmarksPlugin())
    
    # Run the microkernel platform CLI
    kernel.run()

if __name__ == "__main__":
    main()
