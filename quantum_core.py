import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass

@dataclass
class QuantumGate:
    name: str  # "H", "CNOT", "RX", etc.
    qubits: List[int]
    params: Optional[List[float]] = None

class QuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
    
    def h(self, qubit: int):
        self.gates.append(QuantumGate("H", [qubit]))
    
    def cnot(self, control: int, target: int):
        self.gates.append(QuantumGate("CNOT", [control, target]))
    
    def rx(self, qubit: int, theta: float):
        self.gates.append(QuantumGate("RX", [qubit], [theta]))
