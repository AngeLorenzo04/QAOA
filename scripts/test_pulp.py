import sys
import os
import pulp
import networkx as nx
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.data.exact_maxcut_solver import find_exact_maxcut_ilp

# Assumes run from the root of the project
with open("data/graphs/graph_n32_d0.10_id3_seed874052.gpickle", "rb") as f:
    graph = pickle.load(f)

print("Nodes:", graph.nodes())
print("Edges:", graph.edges())
res = find_exact_maxcut_ilp(graph)
print(res)
