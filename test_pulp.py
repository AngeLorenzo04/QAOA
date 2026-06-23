import pulp
import networkx as nx
from src.data.exact_maxcut_solver import find_exact_maxcut_ilp
import pickle

with open("data/graphs/graph_n32_d0.10_id3_seed874052.gpickle", "rb") as f:
    graph = pickle.load(f)

print("Nodes:", graph.nodes())
print("Edges:", graph.edges())
res = find_exact_maxcut_ilp(graph)
print(res)
