# quantum_sim/core/evolution.py
import numpy as np
from quantum_ops import rk4_step
from basis import initialize_restricted_rho
from hamiltonian import generate_hamiltonian_with_selective_k
from observables import construct_observable_in_restricted_basis

def evolve_system_with_selective_k(basis_states, state_to_idx, n_qubits, dt, t_max, qubits_to_excite=None, 
                                   J_max=1.0, k_pattern=None, E_site=0.0, initial_rho=None):
    """
    Evolve the system with selective K coupling.
    If initial_rho is provided, use it as the initial density matrix;
    otherwise, use the classical initialization (using qubits_to_excite).
    """
    # Initialize the density matrix
    if initial_rho is None:
        rho = initialize_restricted_rho(basis_states, state_to_idx, n_qubits, qubits_to_excite)
    else:
        rho = initial_rho
    
    # Generate the Hamiltonian
    H, eigenvalues = generate_hamiltonian_with_selective_k(
        basis_states, state_to_idx, n_qubits, J_max, k_pattern, E_site
    )
    
    # Create projector operators for each qubit
    projectors = [
        construct_observable_in_restricted_basis(basis_states, state_to_idx, n_qubits, q, 'projector_1') 
        for q in range(n_qubits)
    ]
    
    # Set up time points for evolution
    tpoints = np.arange(0, t_max + dt, dt)
    probabilities_1 = np.zeros((len(tpoints), n_qubits))

    # Calculate initial excitation probabilities
    for q in range(n_qubits):
        probabilities_1[0, q] = np.real(np.trace(rho @ projectors[q]))
    
    # Evolve the system step by step
    for t_idx in range(1, len(tpoints)):
        rho = rk4_step(H, rho, dt)
        for q in range(n_qubits):
            probabilities_1[t_idx, q] = np.real(np.trace(rho @ projectors[q]))
    
    return probabilities_1, tpoints, eigenvalues