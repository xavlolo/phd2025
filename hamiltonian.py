# quantum_sim/core/hamiltonian.py
import numpy as np

def generate_hamiltonian_with_selective_k(basis_states, state_to_idx, n_qubits, J_max=1.0, 
                                          k_pattern=None, E_site=0.0):
    """
    Generate Hamiltonian matrix for the quantum system with selective K couplings.
    """
    n_states = len(basis_states)
    H = np.zeros((n_states, n_states), dtype=complex)
    
    # Identify top and bottom qubits
    top_qubits = list(range(0, n_qubits, 2))
    bottom_qubits = list(range(1, n_qubits, 2))
    n_top_qubits = len(top_qubits)
    
    # Default k_pattern if none provided
    if k_pattern is None:
        k_pattern = {(t, b): 0.5 for t, b in zip(top_qubits, bottom_qubits)}
    
    # Calculate J coupling values
    J_values, J_0 = calculate_j_couplings(top_qubits, n_top_qubits, J_max)
    
    # Log J values
    print("\nJ values directly from paper's formula:")
    print(f"J_0 = {J_0:.6f} (base coupling)")
    for (t1, t2), j_val in J_values.items():
        print(f"J({t1},{t2}) = {j_val:.6f} (scaled: {j_val/2:.6f})")
    
    # Calculate theoretical eigenvalues
    eigenvalues_theoretical = calculate_theoretical_eigenvalues(n_top_qubits, J_0)
    
    # Construct the Hamiltonian matrix
    for idx1, state1 in enumerate(basis_states):
        # Diagonal elements: on-site energy
        onsite_energy = E_site * sum(state1)
        H[idx1, idx1] += onsite_energy
        
        # Diagonal elements: ZZ interactions
        for t, b in zip(top_qubits, bottom_qubits):
            k_value = k_pattern.get((t, b), 0.0)
            if state1[t] == state1[b]:
                H[idx1, idx1] += k_value
            else:
                H[idx1, idx1] -= k_value
        
        # Off-diagonal elements: XX+YY interactions
        for i in range(n_top_qubits - 1):
            t1 = top_qubits[i]
            t2 = top_qubits[i + 1]
            if state1[t1] != state1[t2]:
                state2 = list(state1)
                state2[t1] = 1 - state2[t1]
                state2[t2] = 1 - state2[t2]
                state2_tuple = tuple(state2)
                if state2_tuple in state_to_idx:
                    idx2 = state_to_idx[state2_tuple]
                    j_val = J_values.get((t1, t2), 0.0) / 2
                    H[idx1, idx2] += j_val
                    H[idx2, idx1] += j_val
    
    eigenvalues = np.sort(np.real(np.linalg.eigvals(H)))
    print("\nEigenvalues from simulation:")
    for i, ev in enumerate(eigenvalues[:8]):
        print(f"E{i} = {ev:.6f}")
    
    return H, eigenvalues

def calculate_j_couplings(top_qubits, n_top_qubits, J_max):
    """Calculate J coupling values according to perfect state transfer formulas"""
    if n_top_qubits > 1:
        N = n_top_qubits
        if N % 2 == 0:  # Even number of top qubits
            J_0 = 2 * J_max / N
        else:  # Odd number of top qubits
            J_0 = J_max / np.sqrt((N**2/4) - (1/4))
        
        J_values = {}
        for i in range(n_top_qubits - 1):
            t1 = top_qubits[i]
            t2 = top_qubits[i + 1]
            position_i = i + 1
            J_values[(t1, t2)] = J_0 * np.sqrt(position_i * (n_top_qubits - position_i))
    else:
        J_values = {}
        J_0 = 0
    
    return J_values, J_0

def calculate_theoretical_eigenvalues(n_top_qubits, J_0):
    """Calculate theoretical eigenvalues for an isolated spin chain"""
    eigenvalues = []
    if n_top_qubits > 1:
        for k in range(1, n_top_qubits + 1):
            eigenvalue = -2 * J_0 * np.cos(k * np.pi / (n_top_qubits + 1))
            eigenvalues.append(eigenvalue)
        eigenvalues.sort()
        print("\nTheoretical eigenvalues (for an isolated chain):")
        for i, ev in enumerate(eigenvalues[:8]):
            print(f"E{i} = {ev:.6f}")
    
    return eigenvalues