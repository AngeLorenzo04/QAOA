from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit

class MixerStrategy(ABC):
    """
    Abstract base class for QAOA Mixer strategies.
    """
    @abstractmethod
    def apply_initial_state(self, circuit: QuantumCircuit, num_qubits: int) -> None:
        """
        Applies the initial state preparation onto the circuit.
        """
        pass

    @abstractmethod
    def apply_mixer(self, circuit: QuantumCircuit, beta: float, num_qubits: int) -> None:
        """
        Applies the mixer Hamiltonian evolution e^(-i * beta * H_M).
        """
        pass

class StandardMixer(MixerStrategy):
    """
    Standard Transverse-Field Mixer: H_M = sum_i X_i.
    """
    def apply_initial_state(self, circuit: QuantumCircuit, num_qubits: int) -> None:
        # Initial state is the uniform superposition |+>^n
        circuit.h(range(num_qubits))

    def apply_mixer(self, circuit: QuantumCircuit, beta: float, num_qubits: int) -> None:
        # e^(-i * beta * sum X_i) = prod e^(-i * beta * X_i) = prod RX(2 * beta)_i
        circuit.rx(2 * beta, range(num_qubits))


class GroverMixer(MixerStrategy):
    """
    Grover Mixer: H_M = |s><s| where |s> is the uniform superposition state.
    The evolution is U_M(beta) = e^(-i * beta * |s><s|).
    """
    def apply_initial_state(self, circuit: QuantumCircuit, num_qubits: int) -> None:
        # Initial state is the uniform superposition |+>^n
        circuit.h(range(num_qubits))

    def apply_mixer(self, circuit: QuantumCircuit, beta: float, num_qubits: int) -> None:
        # e^(-i * beta * |s><s|) can be implemented by applying H^n,
        # then a phase shift e^(-i * beta) to the state |0..0>, and H^n again.
        
        circuit.h(range(num_qubits))
        
        # Apply phase e^(-i * beta) to |0..0>
        circuit.x(range(num_qubits))
        
        if num_qubits > 1:
            # Multi-Controlled Phase gate adds phase when all control and target qubits are 1
            # We want to apply it when the state is |1..1> (which corresponds to |0..0> before X gates)
            controls = list(range(num_qubits - 1))
            target = num_qubits - 1
            circuit.mcp(-beta, controls, target)
        else:
            # If there's only 1 qubit, it's just a regular Phase gate
            circuit.p(-beta, 0)
            
        circuit.x(range(num_qubits))
        circuit.h(range(num_qubits))

def get_mixer(mixer_name: str) -> MixerStrategy:
    """
    Factory function to get the appropriate mixer strategy.
    """
    mixer_name = mixer_name.lower().strip()
    if mixer_name == 'standard':
        return StandardMixer()
    elif mixer_name == 'grover':
        return GroverMixer()
    else:
        raise ValueError(f"Unknown mixer type: '{mixer_name}'. Supported types are 'standard' and 'grover'.")
