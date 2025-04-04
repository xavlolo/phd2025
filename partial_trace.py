#!/usr/bin/env python3
"""
Module for calculating partial traces and analyzing reduced density matrices
in quantum spin network simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
from quantum_ops import rk4_step

def partial_trace(rho, basis_states, qubits_to_trace_out, n_qubits):
    """
    Calculate the partial trace of the density matrix by tracing out specified qubits.
    
    Parameters:
    - rho: Full density matrix
    - basis_states: List of all basis states in the computational basis
    - qubits_to_trace_out: List of qubit indices to trace out
    - n_qubits: Total number of qubits in the system
    
    Returns:
    - Reduced density matrix after tracing out specified qubits
    - Dictionary mapping indices to subsystem states
    """
    # Determine which qubits to keep
    qubits_to_keep = sorted([q for q in range(n_qubits) if q not in qubits_to_trace_out])
    
    # Get the dimension of the subsystem (2^m where m is the number of qubits kept)
    subsystem_dim = 2**len(qubits_to_keep)
    
    # Initialize the reduced density matrix
    reduced_rho = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
    
    # Create a mapping of basis states to indices for the reduced system
    # These will be the binary representations of states for the qubits we keep
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

def analyze_reduced_density_matrix(basis_states, state_to_idx, n_qubits, dt, t_max, 
                                  initial_rho, H, qubits_to_trace_out=None, 
                                  sample_times=None, plot=True):
    """
    Analyze the time evolution of the reduced density matrix.
    
    Parameters:
    - basis_states: List of basis states
    - state_to_idx: Mapping from states to indices
    - n_qubits: Number of qubits
    - dt: Time step
    - t_max: Maximum simulation time
    - initial_rho: Initial density matrix
    - H: Hamiltonian
    - qubits_to_trace_out: List of qubits to trace out (default: [0, 2, 4] - top qubits)
    - sample_times: List of times at which to sample the reduced density matrix
                   (default: [0, t_max/4, t_max/2, 3*t_max/4, t_max])
    - plot: Whether to generate plots (default: True)
    
    Returns:
    - Dictionary containing analysis results:
        - 'times': List of sampled times
        - 'reduced_rhos': List of reduced density matrices at each time
        - 'state_maps': Mapping from indices to states
        - 'state_probabilities': Probabilities of each state at each time
    """
    if qubits_to_trace_out is None:
        qubits_to_trace_out = [0, 2, 4]  # Default: trace out top qubits
    
    if sample_times is None:
        sample_times = [0, t_max/4, t_max/2, 3*t_max/4, t_max]
    
    # Set up time points
    tpoints = np.arange(0, t_max + dt, dt)
    
    results = {
        'times': [],
        'reduced_rhos': [],
        'state_maps': [],
        'state_probabilities': []
    }
    
    # Initialize density matrix
    rho = initial_rho.copy()
    
    # Process each time point
    for t_idx, t in enumerate(tpoints):
        # Check if current time is in sample_times
        if any(abs(t - sample_t) < dt/2 for sample_t in sample_times):
            # Calculate the reduced density matrix
            reduced_rho, state_map = partial_trace(rho, basis_states, qubits_to_trace_out, n_qubits)
            
            # Extract probabilities (diagonal elements)
            state_probs = np.real(np.diag(reduced_rho))
            
            # Store results
            results['times'].append(t)
            results['reduced_rhos'].append(reduced_rho)
            results['state_maps'].append(state_map)
            results['state_probabilities'].append(state_probs)
            
            print(f"\nTime t = {t:.2f}:")
            print("Reduced density matrix:")
            print(np.round(reduced_rho, 3))
            print("State probabilities:")
            for idx, prob in enumerate(state_probs):
                if prob > 0.01:  # Only show states with non-negligible probability
                    print(f"State |{state_map[idx]}⟩: {prob:.4f}")
        
        # Evolve density matrix to next time step (if not at the end)
        if t_idx < len(tpoints) - 1:
            rho = rk4_step(H, rho, dt)
    
    # Create visualizations if requested
    if plot:
        # Plot the evolution of state probabilities
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all states from the first state map (should be the same for all times)
        state_map = results['state_maps'][0]
        n_states = len(state_map)
        
        # Extract probabilities for each state over time
        for state_idx in range(n_states):
            state_label = state_map[state_idx]
            probs = [results['state_probabilities'][t_idx][state_idx] for t_idx in range(len(results['times']))]
            
            # Only plot states that reach a significant probability at any point
            if max(probs) > 0.05:
                ax.plot(results['times'], probs, 'o-', label=f"|{state_label}⟩")
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.set_title('Evolution of Reduced Density Matrix State Probabilities')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('reduced_density_matrix_evolution.png', dpi=300)
        plt.show()
        
        # Create a stacked bar plot of state probabilities at each sample time
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Only include states that reach a significant probability
        significant_states = []
        for state_idx in range(n_states):
            max_prob = max([results['state_probabilities'][t_idx][state_idx] for t_idx in range(len(results['times']))])
            if max_prob > 0.05:
                significant_states.append(state_idx)
        
        # Set up bar positions
        x = np.arange(len(results['times']))
        width = 0.7
        
        # Set up colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(significant_states)))
        
        # Create bars
        bottom = np.zeros(len(results['times']))
        for i, state_idx in enumerate(significant_states):
            state_label = state_map[state_idx]
            probs = [results['state_probabilities'][t_idx][state_idx] for t_idx in range(len(results['times']))]
            ax.bar(x, probs, width, bottom=bottom, label=f"|{state_label}⟩", color=colors[i])
            bottom += probs
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.set_title('State Composition of Reduced Density Matrix Over Time')
        ax.set_xticks(x)
        ax.set_xticklabels([f't = {t:.2f}' for t in results['times']])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('reduced_density_matrix_composition.png', dpi=300)
        plt.show()
        
        # Create heatmaps of the reduced density matrix at each time
        for t_idx, t in enumerate(results['times']):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            reduced_rho = results['reduced_rhos'][t_idx]
            abs_reduced_rho = np.abs(reduced_rho)
            
            im = ax.imshow(abs_reduced_rho, cmap='viridis')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Absolute Value', rotation=-90, va="bottom")
            
            # Set ticks and labels
            state_labels = [f"|{state_map[i]}⟩" for i in range(n_states)]
            ax.set_xticks(np.arange(n_states))
            ax.set_yticks(np.arange(n_states))
            ax.set_xticklabels(state_labels)
            ax.set_yticklabels(state_labels)
            
            # Rotate the x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add title
            ax.set_title(f"Reduced Density Matrix (abs values) at t = {t:.2f}")
            
            # Add text annotations with values
            for i in range(n_states):
                for j in range(n_states):
                    value = abs_reduced_rho[i, j]
                    if value > 0.05:  # Only annotate significant values
                        text = ax.text(j, i, f"{value:.2f}",
                                      ha="center", va="center", 
                                      color="white" if value > 0.5 else "black")
            
            plt.tight_layout()
            plt.savefig(f'reduced_density_matrix_t{t:.2f}.png', dpi=300)
            plt.show()
    
    return results

def check_purity_entropy(reduced_rho):
    """
    Calculate the purity and von Neumann entropy of a reduced density matrix.
    
    Parameters:
    - reduced_rho: Reduced density matrix
    
    Returns:
    - purity: Tr(ρ²)
    - entropy: -Tr(ρ log ρ)
    """
    # Calculate purity: Tr(ρ²)
    purity = np.real(np.trace(reduced_rho @ reduced_rho))
    
    # Calculate eigenvalues for entropy
    eigenvalues = np.linalg.eigvalsh(reduced_rho)
    
    # Filter out very small eigenvalues to avoid log(0) issues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Calculate von Neumann entropy: S = -Tr(ρ log ρ) = -∑ λ log λ
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return purity, entropy

def analyze_entanglement_over_time(basis_states, state_to_idx, n_qubits, dt, t_max, 
                                  initial_rho, H, qubits_to_trace_out=None, 
                                  num_samples=50):
    """
    Analyze the entanglement between subsystems over time using purity and entropy.
    
    Parameters:
    - basis_states, state_to_idx, n_qubits: System description
    - dt, t_max: Time parameters
    - initial_rho: Initial density matrix
    - H: Hamiltonian
    - qubits_to_trace_out: Qubits to trace out (default: top qubits [0, 2, 4])
    - num_samples: Number of time points to sample
    
    Returns:
    - Dictionary with time series data for purity and entropy
    """
    if qubits_to_trace_out is None:
        qubits_to_trace_out = [0, 2, 4]  # Default: trace out top qubits
    
    # Sample times evenly
    sample_indices = np.linspace(0, int(t_max/dt), num_samples, dtype=int)
    tpoints = np.arange(0, t_max + dt, dt)
    sample_times = [tpoints[i] for i in sample_indices if i < len(tpoints)]
    
    results = {
        'times': sample_times,
        'purity': [],
        'entropy': []
    }
    
    # Initialize density matrix
    rho = initial_rho.copy()
    
    # Loop through all time steps
    for t_idx, t in enumerate(tpoints):
        # Check if this is a sample point
        if t_idx in sample_indices and t_idx < len(tpoints):
            # Calculate reduced density matrix
            reduced_rho, _ = partial_trace(rho, basis_states, qubits_to_trace_out, n_qubits)
            
            # Calculate purity and entropy
            purity, entropy = check_purity_entropy(reduced_rho)
            
            # Store results
            results['purity'].append(purity)
            results['entropy'].append(entropy)
            
            print(f"Time t = {t:.2f}: Purity = {purity:.4f}, Entropy = {entropy:.4f}")
        
        # Evolve to next time step
        if t_idx < len(tpoints) - 1:
            rho = rk4_step(H, rho, dt)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Purity plot
    ax1.plot(results['times'], results['purity'], 'o-', color='blue')
    ax1.set_ylabel('Purity: Tr(ρ²)')
    ax1.set_title('Purity of Reduced Density Matrix Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Pure State')
    ax1.axhline(y=1/8, color='green', linestyle='--', alpha=0.5, label='Maximally Mixed')
    ax1.legend()
    
    # Entropy plot
    ax2.plot(results['times'], results['entropy'], 'o-', color='purple')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('von Neumann Entropy')
    ax2.set_title('Entropy of Reduced Density Matrix Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add reference lines
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Pure State')
    ax2.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='Maximally Mixed')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('entanglement_measures_over_time.png', dpi=300)
    plt.show()
    
    return results

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    from basis import generate_selective_basis, initialize_superposition_rho
    from hamiltonian import generate_hamiltonian_with_selective_k
    
    # Test parameters
    n_qubits = 6
    dt = 0.01
    t_max = 50
    
    # Generate basis
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits, 
        top_excitations=True, 
        bottom_excitations=True
    )
    
    # Define superposition
    superposition_dict = {
        (1, 1, 0, 0, 0, 0): 1.0 + 0j,
        (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,
        (1, 0, 0, 0, 0, 1): 0.3 - 0.2j
    }
    
    # Initialize density matrix
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # Generate Hamiltonian
    k_pattern = {(0, 1): 1.5, (2, 3): 1.0, (4, 5): 0.5}
    H, _ = generate_hamiltonian_with_selective_k(
        basis_states, state_to_idx, n_qubits, 
        J_max=1.0, k_pattern=k_pattern, E_site=0.0
    )
    
    # Analyze reduced density matrix
    analyze_reduced_density_matrix(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        initial_rho, H
    )
    
    # Analyze entanglement
    analyze_entanglement_over_time(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        initial_rho, H
    )