import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_QUBITS = 11

def update_theta(theta, elite_bitstrings, learning_rate=0.15):
    """
    Update theta based on elite bitstrings.
    If elite candidates have 1 more often at a position, increase theta.
    If they have 0 more often, decrease theta.
    """

    new_theta = theta.copy()

    for i in range(len(theta)):
        bits_at_i = [int(bitstring[i]) for bitstring in elite_bitstrings]
        mean_bit = sum(bits_at_i) / len(bits_at_i)

        if mean_bit > 0.5:
            new_theta[i] += learning_rate
        elif mean_bit < 0.5:
            new_theta[i] -= learning_rate

        new_theta[i] = max(0.05, min(np.pi - 0.05, new_theta[i]))

    return new_theta


def build_sampling_circuit(theta):
    if len(theta) != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS} theta values, got {len(theta)}")

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    for i, angle in enumerate(theta):
        qc.ry(angle, i)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc

def sample_bitstrings(theta, shots=32):
    qc = build_sampling_circuit(theta)

    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    return counts

def get_unique_bitstrings(counts):
    return list(counts.keys())

if __name__ == "__main__":
    theta = np.ones(N_QUBITS) * (np.pi / 4)

    counts = sample_bitstrings(theta, shots=32)

    print("Sampled counts:")
    for bitstring, count in counts.items():
        print(bitstring, "->", count)

    print("\nUnique sampled candidates:")
    print(get_unique_bitstrings(counts))