from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import cupy as cp

@dataclass
class QuantumGate:
    name: str
    qubits: List[int]
    params: Optional[List[float]] = None

class QuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates = []
    
    def h(self, qubit: int):
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("H", [qubit]))
    
    def cnot(self, control: int, target: int):
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.gates.append(QuantumGate("CNOT", [control, target]))
    
    def rx(self, qubit: int, theta: float):
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("RX", [qubit], [theta]))
    
    def measure_all(self):
        self.gates.append(QuantumGate("MEASURE", list(range(self.num_qubits))))
    
    def _validate_qubit(self, qubit: int):
        if qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range")
    
    def to_qasm(self) -> str:
        qasm = [f"OPENQASM 2.0;", f"include \"qelib1.inc\";", 
                f"qreg q[{self.num_qubits}];", f"creg c[{self.num_qubits}];"]
        for gate in self.gates:
            if gate.name == "MEASURE":
                qasm.extend([f"measure q[{i}] -> c[{i}];" for i in gate.qubits])
            else:
                params = "" if not gate.params else f"({','.join(map(str, gate.params))})"
                qasm.append(f"{gate.name.lower()}{params} q[{gate.qubits[0]}];")
        return "\n".join(qasm)
