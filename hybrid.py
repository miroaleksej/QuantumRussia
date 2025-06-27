import tensorflow as tf

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit: QuantumCircuit, simulator: StatevectorSimulator):
        super().__init__()
        self.circuit = circuit
        self.simulator = simulator
    
    def call(self, inputs):
        # 1. Загрузка классических данных в квантовую схему
        for i, param in enumerate(inputs):
            self.circuit.rx(i % self.circuit.num_qubits, param)
        
        # 2. Запуск симулятора
        self.simulator.run(self.circuit)
        
        # 3. Возврат ожидаемых значений
        return self.simulator.get_expectation_values()

class HybridModel(tf.keras.Model):
    def __init__(self, quantum_circuit: QuantumCircuit):
        super().__init__()
        self.quantum_layer = QuantumLayer(quantum_circuit)
        self.classical = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, inputs):
        x = self.quantum_layer(inputs)
        return self.classical(x)
