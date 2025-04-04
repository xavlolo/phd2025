# quantum_sim/core/basis.py
import numpy as np
from itertools import chain, combinations

def generate_selective_basis(n_qubits, top_excitations=None, bottom_excitations=None):
    """Generate basis states with selective control over which qubits can be excited
       using a single power set over the union of allowed excitations."""
    # Identify top and bottom qubits
    top_qubits = list(range(0, n_qubits, 2))
    bottom_qubits = list(range(1, n_qubits, 2))

    # Process top_excitations parameter
    if top_excitations is None:
        allowed_top = []
    elif top_excitations is True:
        allowed_top = top_qubits
    else:
        allowed_top = [q for q in top_excitations if q in top_qubits]
        invalid_top = [q for q in top_excitations if q not in top_qubits]
        if invalid_top:
            print(f"Warning: Qubits {invalid_top} are not top qubits and will be ignored")

    # Process bottom_excitations parameter
    if bottom_excitations is None:
        allowed_bottom = []
    elif bottom_excitations is True:
        allowed_bottom = bottom_qubits
    else:
        allowed_bottom = [q for q in bottom_excitations if q in bottom_qubits]
        invalid_bottom = [q for q in bottom_excitations if q not in bottom_qubits]
        if invalid_bottom:
            print(f"Warning: Qubits {invalid_bottom} are not bottom qubits and will be ignored")

    # Combine allowed top and bottom qubits
    allowed = sorted(set(allowed_top).union(set(allowed_bottom)))

    # Generate basis states
    basis_states = []
    for subset in powerset(allowed):
        state = [0] * n_qubits
        for q in subset:
            state[q] = 1
        basis_states.append(tuple(state))

    # Create a mapping from states to their indices
    state_to_idx = {state: idx for idx, state in enumerate(basis_states)}

    print(f"Generated {len(basis_states)} basis states for {n_qubits} qubits")
    print(f"Excitable qubits: {allowed}")

    return basis_states, state_to_idx

def powerset(iterable):
    """Generate all possible subsets of an iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def initialize_restricted_rho(basis_states, state_to_idx, n_qubits, qubits_to_excite=None):
    """Initialize the density matrix in the restricted basis for a single basis state"""
    if qubits_to_excite is None:
        qubits_to_excite = []
    
    state = [0] * n_qubits
    for q in qubits_to_excite:
        state[q] = 1
    state_tuple = tuple(state)
    
    if state_tuple not in state_to_idx:
        raise ValueError(f"State with qubits {qubits_to_excite} excited is not in the restricted basis")
    
    idx = state_to_idx[state_tuple]
    
    n_states = len(basis_states)
    rho = np.zeros((n_states, n_states), dtype=complex)
    rho[idx, idx] = 1.0
    
    return rho

def initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict):
    """
    Initialize the density matrix from a superposition.
    superposition_dict should map basis state tuples (e.g., (0,1,0,0,0,0))
    to their complex amplitudes.
    """
    n_states = len(basis_states)
    psi = np.zeros(n_states, dtype=complex)
    
    for state, amp in superposition_dict.items():
        if state not in state_to_idx:
            raise ValueError(f"State {state} is not in the restricted basis")
        psi[state_to_idx[state]] = amp
    
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("State vector has zero norm!")
    psi /= norm
    
    return np.outer(psi, np.conjugate(psi))