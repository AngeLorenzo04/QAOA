from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler

qc = QuantumCircuit(2)
qc.h([0, 1])
qc.measure_all()

sampler = Sampler()
job = sampler.run(qc)
result = job.result()
dist = result.quasi_dists[0]

for state, prob in dist.items():
    print(type(state), state, prob)
