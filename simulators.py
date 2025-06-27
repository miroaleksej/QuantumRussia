import numpy as np
import cupy as cp
from numba import cuda
from typing import Dict

class StatevectorSimulator:
    def __init__(self, num_qubits: int, use_gpu: bool = False):
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
    
    def run(self, circuit: QuantumCircuit):
        for gate in circuit.gates:
            if gate.name == "H":
                self._apply_hadamard(gate.qubits[0])
            elif gate.name == "CNOT":
                self._apply_cnot(gate.qubits[0], gate.qubits[1])
            elif gate.name == "RX":
                self._apply_rx(gate.qubits[0], gate.params[0])
    
    def _apply_hadamard(self, qubit: int):
        if self.use_gpu:
            self._hadamard_gpu(qubit)
        else:
            self._hadamard_cpu(qubit)
    
    def _hadamard_cpu(self, qubit: int):
        stride = 2**qubit
        for i in range(0, len(self.state), 2**(qubit+1)):
            for j in range(i, i + stride):
                a, b = self.state[j], self.state[j + stride]
                self.state[j] = (a + b) / np.sqrt(2)
                self.state[j + stride] = (a - b) / np.sqrt(2)
    
    @staticmethod
    @cuda.jit(device=True)
    def _hadamard_kernel(state, qubit_mask, size):
        i = cuda.grid(1)
        if i < size:
            j = i ^ qubit_mask
            if j > i:
                a = state[i]
                b = state[j]
                state[i] = (a + b) / 1.41421356237  # 1/sqrt(2)
                state[j] = (a - b) / 1.41421356237
    
    def _hadamard_gpu(self, qubit: int):
        qubit_mask = 1 << qubit
        size = len(self.state)
        threads = 256
        blocks = (size + threads - 1) // threads
        self._hadamard_kernel[blocks, threads](self.state, qubit_mask, size)
    
    def get_expectation_values(self):
        if self.use_gpu:
            return cp.abs(self.state.get())**2
        return np.abs(self.state)**2

class NoisySimulator(StatevectorSimulator):
    def __init__(self, num_qubits: int, noise_model: Dict):
        super().__init__(num_qubits)
        self.noise_model = noise_model
    
    def apply_noise(self):
        if "depolarizing" in self.noise_model:
            p = self.noise_model["depolarizing"]
            if self.use_gpu:
                noise = cp.random.random(len(self.state)) < p
                self.state[noise] = 0
            else:
                noise = np.random.random(len(self.state)) < p
                self.state[noise] = 0
