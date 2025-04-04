# quantum_sim/analysis/fidelity.py
import numpy as np
from scipy.linalg import expm
from quantum_ops import rk4_step

def calculate_standard_fidelity(basis_states, state_to_idx, initial_state_dict, H, final_time, desired_state_dict=None):
    """
    Calculate the standard quantum fidelity between the evolved initial state and a desired state.
    
    F(t) = |<Ψ_des|e^(-iHt/ħ)|Ψ(0)>|^2
    
    Parameters:
    -----------
    basis_states : list
        List of all basis states in the computational basis
    state_to_idx : dict
        Mapping from basis states to their indices
    initial_state_dict : dict
        Dictionary mapping basis state tuples to their amplitudes in the initial state
    H : numpy.ndarray
        The system Hamiltonian
    final_time : float
        The time at which to evaluate the fidelity
    desired_state_dict : dict, optional
        Dictionary mapping basis state tuples to their amplitudes in the desired state.
        If None, the initial state is used.
        
    Returns:
    --------
    fidelity : float
        The quantum fidelity between the evolved state and the desired state
    psi_final : numpy.ndarray
        The evolved state vector at time final_time
    """
    # Construct the initial state vector
    n_states = len(basis_states)
    psi_initial = np.zeros(n_states, dtype=complex)
    
    for state, amp in initial_state_dict.items():
        psi_initial[state_to_idx[state]] = amp
    
    # Normalize the initial state
    psi_initial /= np.linalg.norm(psi_initial)
    
    # Perform time evolution using matrix exponentiation
    psi_final = expm(-1j * H * final_time) @ psi_initial
    
    # Construct the desired state vector (if provided)
    if desired_state_dict is None:
        # If no desired state is provided, use the initial state
        psi_desired = psi_initial
    else:
        psi_desired = np.zeros(n_states, dtype=complex)
        for state, amp in desired_state_dict.items():
            psi_desired[state_to_idx[state]] = amp
        psi_desired /= np.linalg.norm(psi_desired)
    
    # Calculate the fidelity as |<Ψ_des|Ψ(t)>|^2
    overlap = np.vdot(psi_desired, psi_final)
    fidelity = np.abs(overlap)**2
    
    return fidelity, psi_final

def partial_trace(rho, basis_states, qubits_to_trace_out, n_qubits):
    """
    Calculate the partial trace of the density matrix by tracing out specified qubits.
    
    Parameters:
    -----------
    rho : numpy.ndarray
        Full density matrix
    basis_states : list
        List of all basis states in the computational basis
    qubits_to_trace_out : list
        List of qubit indices to trace out
    n_qubits : int
        Total number of qubits in the system
    
    Returns:
    --------
    reduced_rho : numpy.ndarray
        Reduced density matrix after tracing out specified qubits
    index_to_state : dict
        Dictionary mapping indices to subsystem states
    """
    # Determine which qubits to keep
    qubits_to_keep = sorted([q for q in range(n_qubits) if q not in qubits_to_trace_out])
    
    # Get the dimension of the subsystem
    subsystem_dim = 2**len(qubits_to_keep)
    
    # Initialize the reduced density matrix
    reduced_rho = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
    
    # Create a mapping of basis states to indices for the reduced system
    subsystem_basis_map = {}
    for i in range(subsystem_dim):
        # Convert integer to binary representation
        binary = format(i, f'0{len(qubits_to_keep)}b')
        # Create tuple of binary digits
        state_tuple = tuple(int(binary[j]) for j in range(len(qubits_to_keep)))
        subsystem_basis_map[state_tuple] = i
    
    # Store the reverse mapping (index to string representation)
    index_to_state = {i: ''.join(map(str, state_tuple)) for state_tuple, i in subsystem_basis_map.items()}
    
    # Loop through all combinations of basis states in the full space
    for i, basis_i in enumerate(basis_states):
        for j, basis_j in enumerate(basis_states):
            # Check if states match for the qubits being traced out
            match = True
            for q in qubits_to_trace_out:
                if basis_i[q] != basis_j[q]:
                    match = False
                    break
            
            if match:
                # Extract the states of the qubits we're keeping
                kept_i = tuple(basis_i[q] for q in qubits_to_keep)
                kept_j = tuple(basis_j[q] for q in qubits_to_keep)
                
                # Find the corresponding indices in the reduced matrix
                subsys_i = subsystem_basis_map[kept_i]
                subsys_j = subsystem_basis_map[kept_j]
                
                # Add the contribution to the reduced density matrix
                reduced_rho[subsys_i, subsys_j] += rho[i, j]
    
    return reduced_rho, index_to_state

def calculate_simplified_fidelity(probabilities_1, tpoints, initial_superposition_dict, basis_states, state_to_idx, bottom_qubits):
    """
    Calculate a simplified version of the fidelity between initial and final qubit states
    based primarily on excitation probabilities.
    
    Parameters:
    -----------
    probabilities_1 : numpy.ndarray
        Array of excitation probabilities for each qubit over time
    tpoints : numpy.ndarray
        Array of time points
    initial_superposition_dict : dict
        Dictionary mapping basis state tuples to their amplitudes in the initial state
    basis_states : list
        List of all basis states in the computational basis
    state_to_idx : dict
        Mapping from basis states to their indices
    bottom_qubits : list
        List of indices for bottom qubits
    
    Returns:
    --------
    fidelities : dict
        Dictionary mapping qubit pairs (q1, q2) to their simplified fidelity
    initial_probs : dict
        Dictionary mapping qubits to their initial excitation probabilities
    final_probs : dict
        Dictionary mapping qubits to their final excitation probabilities
    """
    # Calculate initial probabilities for each bottom qubit from the superposition
    initial_probs = {}
    total_prob = sum(np.abs(amp)**2 for amp in initial_superposition_dict.values())
    
    for q in bottom_qubits:
        # Sum up probabilities for all basis states where qubit q is excited
        q_prob = 0
        for state, amp in initial_superposition_dict.items():
            if state[q] == 1:
                q_prob += np.abs(amp)**2
        initial_probs[q] = q_prob / total_prob
    
    # Get final probabilities (at t_max)
    final_probs = {}
    for q in bottom_qubits:
        final_probs[q] = probabilities_1[-1, q]
    
    # Calculate simplified fidelity for all pairs of bottom qubits
    fidelities = {}
    
    for q1 in bottom_qubits:
        p1 = initial_probs[q1]  # Probability of q1 being excited at t=0
        
        for q2 in bottom_qubits:
            p2 = final_probs[q2]  # Probability of q2 being excited at t=t_max
            
            # Simple fidelity calculation based on probability overlap
            fidelity = 1 - abs(p1 - p2)
            fidelities[(q1, q2)] = fidelity
    
    return fidelities, initial_probs, final_probs

def analyze_fidelity(basis_states, state_to_idx, n_qubits, dt, t_max, superposition_dict, H, k_pattern=None):
    """
    Analyze quantum fidelity between initial and final states.
    
    Parameters:
    -----------
    basis_states : list
        List of all basis states in the computational basis
    state_to_idx : dict
        Mapping from basis states to their indices
    n_qubits : int
        Number of qubits in the system
    dt : float
        Time step for evolution
    t_max : float
        Maximum simulation time
    superposition_dict : dict
        Dictionary mapping basis state tuples to their amplitudes
    H : numpy.ndarray
        The system Hamiltonian
    k_pattern : dict, optional
        Dictionary mapping (top, bottom) qubit pairs to their coupling strengths
        
    Returns:
    --------
    dict
        Dictionary containing fidelity analysis results
    """
    import matplotlib.pyplot as plt
    
    # Identify top and bottom qubits
    top_qubits = list(range(0, n_qubits, 2))
    bottom_qubits = list(range(1, n_qubits, 2))
    
    # Initialize the density matrix
    from basis import initialize_superposition_rho
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # Evolve the system to the final time
    from evolution import evolve_system_with_selective_k
    probabilities_1, times, _ = evolve_system_with_selective_k(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        J_max=1.0, k_pattern=k_pattern, E_site=0.0, initial_rho=initial_rho
    )
    
    # Get the final density matrix by re-evolving
    final_rho = initial_rho.copy()
    for t_idx in range(1, len(times)):
        final_rho = rk4_step(H, final_rho, dt)
    
    # Calculate standard fidelity for transfer to each bottom qubit
    print("Calculating standard quantum fidelity:")
    std_fidelities = {}
    for q in bottom_qubits:
        # Define the desired state with qubit 0 and q excited
        desired_state = [0] * n_qubits
        desired_state[0] = 1  # Top qubit 0 always excited
        desired_state[q] = 1  # Bottom qubit q excited
        desired_state_dict = {tuple(desired_state): 1.0}
        
        # Calculate fidelity
        fidelity, _ = calculate_standard_fidelity(
            basis_states, state_to_idx, superposition_dict, H, t_max, desired_state_dict
        )
        std_fidelities[q] = fidelity
        print(f"Fidelity from initial superposition to state with qubits 0 and {q} excited: {fidelity:.4f}")
    
    # Calculate simplified fidelity
    print("\nCalculating simplified fidelity between initial and final qubit states:")
    simpl_fidelities, initial_probs, final_probs = calculate_simplified_fidelity(
        probabilities_1, times, superposition_dict, basis_states, state_to_idx, bottom_qubits
    )
    
    # Print the results
    print("\nInitial excitation probabilities:")
    for q, prob in initial_probs.items():
        print(f"Bottom Qubit {q}: {prob:.4f}")
    
    print("\nFinal excitation probabilities:")
    for q, prob in final_probs.items():
        print(f"Bottom Qubit {q}: {prob:.4f}")
    
    print("\nFidelity between initial and final qubit states:")
    print(f"{'Initial Qubit':<15} {'Final Qubit':<15} {'Fidelity':<10}")
    print("-" * 40)
    
    for (q1, q2), fidelity in simpl_fidelities.items():
        print(f"Bottom Qubit {q1:<5} Bottom Qubit {q2:<5} {fidelity:.4f}")
    
    # Calculate partial trace over top qubits
    print("\nCalculating partial trace over top qubits (0, 2, 4):")
    reduced_bottom_rho, bottom_state_map = partial_trace(
        final_rho, basis_states, top_qubits, n_qubits
    )
    
    # Verify trace = 1
    trace_value = np.real(np.trace(reduced_bottom_rho))
    print(f"Trace of reduced density matrix: {trace_value:.6f} (should be close to 1.0)")
    
    # Calculate probabilities for different states of bottom qubits (diagonal elements)
    bottom_state_probs = np.real(np.diag(reduced_bottom_rho))
    
    # Print probabilities of each bottom qubit state
    print("\nProbabilities for bottom qubit states:")
    for idx, prob in enumerate(bottom_state_probs):
        state = bottom_state_map[idx]
        print(f"State |{state}⟩: {prob:.4f}")
    
    # Create visualizations
    
    # 1. Fidelity matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    fidelity_matrix = np.zeros((len(bottom_qubits), len(bottom_qubits)))
    
    for i, q1 in enumerate(bottom_qubits):
        for j, q2 in enumerate(bottom_qubits):
            fidelity_matrix[i, j] = simpl_fidelities[(q1, q2)]
    
    im = ax.imshow(fidelity_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Fidelity', rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(bottom_qubits)))
    ax.set_yticks(np.arange(len(bottom_qubits)))
    ax.set_xticklabels([f'Q{q}' for q in bottom_qubits])
    ax.set_yticklabels([f'Q{q}' for q in bottom_qubits])
    
    # Add title and labels
    ax.set_title("Fidelity Between Initial and Final Qubit States")
    ax.set_xlabel("Final Qubit State")
    ax.set_ylabel("Initial Qubit State")
    
    # Add text annotations with the fidelity values
    for i in range(len(bottom_qubits)):
        for j in range(len(bottom_qubits)):
            ax.text(j, i, f"{fidelity_matrix[i, j]:.2f}",
                   ha="center", va="center", color="w" if fidelity_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    plt.savefig("output/fidelity_matrix.png", dpi=300)
    plt.show()
    
    # 2. Bottom qubit state probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort states in binary order
    sorted_indices = sorted(range(len(bottom_state_probs)), key=lambda i: bottom_state_map[i])
    sorted_probs = [bottom_state_probs[i] for i in sorted_indices]
    sorted_states = [bottom_state_map[i] for i in sorted_indices]
    
    bars = ax.bar(range(len(sorted_probs)), sorted_probs)
    
    # Add state labels
    ax.set_xticks(range(len(sorted_probs)))
    ax.set_xticklabels([f"|{state}⟩" for state in sorted_states], rotation=45)
    
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1.0)  # Set y-axis from 0 to 1
    ax.set_title('Bottom Qubits (1,3,5) State Probabilities After Evolution')
    
    # Add text labels on the bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.02:  # Only label bars with non-negligible height
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("output/bottom_qubit_states.png", dpi=300)
    plt.show()
    
    # 3. Comparison of initial vs final probabilities
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for comparison (assuming 3 bottom qubits = 8 possible states)
    all_bottom_states = ['000', '001', '010', '011', '100', '101', '110', '111']
    
    # Get initial probabilities for each bottom state
    total_prob = sum(np.abs(amp)**2 for amp in superposition_dict.values())
    initial_bottom_probs = np.zeros(len(all_bottom_states))
    for state, amp in superposition_dict.items():
        bottom_state = f"{state[1]}{state[3]}{state[5]}"
        idx = int(bottom_state, 2)  # Convert binary string to integer
        initial_bottom_probs[idx] = np.abs(amp)**2/total_prob
    
    # Get final probabilities in the same order
    final_bottom_probs = np.zeros(len(all_bottom_states))
    for idx, state in bottom_state_map.items():
        binary_idx = int(state, 2)  # Convert binary string to integer
        final_bottom_probs[binary_idx] = bottom_state_probs[idx]
    
    # Set the positions for the bars
    x = np.arange(len(all_bottom_states))
    width = 0.35
    
    # Create the bars
    ax.bar(x - width/2, initial_bottom_probs, width, label='Initial')
    ax.bar(x + width/2, final_bottom_probs, width, label='Final')
    
    # Add labels and title
    ax.set_xlabel('Bottom Qubit State')
    ax.set_ylabel('Probability')
    ax.set_title('Comparison of Initial vs Final Bottom Qubit State Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels([f"|{state}⟩" for state in all_bottom_states])
    ax.legend()
    
    # Add text labels on significant bars
    for i, (init_prob, final_prob) in enumerate(zip(initial_bottom_probs, final_bottom_probs)):
        if init_prob > 0.02:
            ax.text(i - width/2, init_prob + 0.02, f'{init_prob:.3f}', ha='center', va='bottom')
        if final_prob > 0.02:
            ax.text(i + width/2, final_prob + 0.02, f'{final_prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("output/bottom_qubit_states_comparison.png", dpi=300)
    plt.show()
    
    return {
        'standard_fidelities': std_fidelities,
        'simplified_fidelities': simpl_fidelities,
        'initial_probabilities': initial_probs,
        'final_probabilities': final_probs,
        'bottom_state_probabilities': dict(zip([bottom_state_map[i] for i in range(len(bottom_state_probs))], bottom_state_probs)),
        'reduced_density_matrix': reduced_bottom_rho
    }