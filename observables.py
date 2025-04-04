# quantum_sim/core/observables.py
import numpy as np

def construct_observable_in_restricted_basis(basis_states, state_to_idx, n_qubits, qubit_index, operator_type='sigma_z'):
    """Construct an observable operator in the restricted basis"""
    n_states = len(basis_states)
    op = np.zeros((n_states, n_states), dtype=complex)
    
    for idx, state in enumerate(basis_states):
        if operator_type == 'sigma_z':
            op[idx, idx] = 1 - 2*state[qubit_index]
        elif operator_type == 'projector_1':
            op[idx, idx] = state[qubit_index]
    
    return op