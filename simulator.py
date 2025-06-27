import cupy as cp
import numpy as np
from numba import cuda

class StatevectorSimulator:
    def __init__(self, num_qubits: int, use_gpu: bool = True):
        self.num_qubits = num_qubits
        self.use_gpu = use_gpu
        self.state = self._init_state()
    
    def _init_state(self):
        size = 2**self.num_qubits
        if self.use_gpu:
            state = cp.zeros(size, dtype=cp.complex128)
            state[0] = 1.0
        else:
            state = np.zeros(size, dtype=np.complex128)
            state[0] = 1.0
        return state
    
    def apply_gate(self, gate: QuantumGate):
        if gate.name == "H":
            self._apply_hadamard(gate.qubits[0])
        elif gate.name == "CNOT":
            self._apply_cnot(gate.qubits[0], gate.qubits[1])
    
    @cuda.jit(device=True)
    def _apply_hadamard_gpu(state, qubit):
        # CUDA-ядро для гейта Адамара
        pass
