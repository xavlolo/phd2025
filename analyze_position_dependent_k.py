def analyze_position_dependent_k(plot=True):
    """
    Analyze the effects of position-dependent K-coupling on quantum spin network dynamics.
    """
    # Parameters for the simulation
    n_qubits = 6
    dt = 0.01
    t_max = 50
    J_max = 1.0
    E_site = 0.0
    analyze_qubit = 4
    
    # Results storage
    results = {}
    
    # Generate basis states
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits,
        top_excitations=True,
        bottom_excitations=True
    )
    
    # Define the superposition initial state
    superposition_dict = {
        (1, 1, 0, 0, 0, 0): 1.0 + 0j,
        (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,
        (1, 0, 0, 0, 0, 1): 0.3 - 0.2j
    }
    
    # Initialize the density matrix
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # Define position-dependent K patterns to test
    k_patterns = {
        "Uniform K=1.0": {
            (0, 1): 1.0, (2, 3): 1.0, (4, 5): 1.0
        },
        "Strong at Beginning": {
            (0, 1): 1.5, (2, 3): 0.5, (4, 5): 0.5
        },
        "Strong in Middle": {
            (0, 1): 0.5, (2, 3): 1.5, (4, 5): 0.5
        },
        "Strong at End": {
            (0, 1): 0.5, (2, 3): 0.5, (4, 5): 1.5
        },
        "Linearly Increasing": {
            (0, 1): 0.5, (2, 3): 1.0, (4, 5): 1.5
        },
        "Linearly Decreasing": {
            (0, 1): 1.5, (2, 3): 1.0, (4, 5): 0.5
        }
    }
    
    # Iterate through different K patterns
    for pattern_name, k_pattern in k_patterns.items():
        # Generate Hamiltonian
        H, eigenvalues = generate_hamiltonian_with_selective_k(
            basis_states, state_to_idx, n_qubits, J_max, k_pattern, E_site
        )
        
        # Evolve the system
        probabilities_1, times, _ = evolve_system_with_selective_k(
            basis_states, state_to_idx, n_qubits, dt, t_max,
            J_max=J_max, k_pattern=k_pattern, E_site=E_site, initial_rho=initial_rho
        )
        
        # Perform FFT analysis
        freqs_pos, abs_fft, peak_freqs, peak_amps = analyze_fft(
            probabilities_1, times, analyze_qubit
        )
        
        # Store results
        results[pattern_name] = {
            'k_pattern': k_pattern,
            'probabilities': probabilities_1,
            'times': times,
            'fft_freqs': freqs_pos,
            'fft_amps': abs_fft,
            'peak_freqs': peak_freqs,
            'peak_amps': peak_amps,
            'eigenvalues': eigenvalues
        }
    
    # Plotting if requested
    if plot:
        fig, axs = plt.subplots(len(k_patterns), 2, figsize=(16, 4*len(k_patterns)))
        fig.suptitle("Position-Dependent K-Coupling Effects", fontsize=16)
        
        for i, (pattern_name, result) in enumerate(results.items()):
            # Dynamics plot
            axs[i, 0].set_title(f'Qubit Dynamics: {pattern_name}')
            for q in range(n_qubits):
                linestyle = '-' if q % 2 == 0 else '--'
                qubit_type = "Top" if q % 2 == 0 else "Bottom"
                axs[i, 0].plot(result['times'], result['probabilities'][:, q], 
                               label=f'{qubit_type} Qubit {q}', 
                               linestyle=linestyle)
            axs[i, 0].set_xlabel('Time')
            axs[i, 0].set_ylabel('Excitation Probability')
            axs[i, 0].legend(loc='best')
            axs[i, 0].grid(True, alpha=0.3)
            
            # FFT plot
            axs[i, 1].set_title(f'FFT Analysis: {pattern_name}')
            axs[i, 1].plot(result['fft_freqs'], result['fft_amps'], color='navy', linewidth=1.5)
            axs[i, 1].set_xlabel('Frequency (Hz)')
            axs[i, 1].set_ylabel('Amplitude')
            axs[i, 1].grid(True, alpha=0.3)
            axs[i, 1].set_xlim(0, min(2, max(result['fft_freqs'])))
            
            # Mark peaks
            for freq, amp in zip(result['peak_freqs'][:3], result['peak_amps'][:3]):
                axs[i, 1].plot(freq, amp, 'ro', markersize=6)
                axs[i, 1].text(freq + 0.02, amp, f'{freq:.4f} Hz', 
                               verticalalignment='center',
                               fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('position_dependent_k_effects.png', dpi=300)
        plt.show()
        
        # Create summary comparison
        # 1. Compare energy spectra
        plt.figure(figsize=(12, 8))
        for pattern_name, result in results.items():
            plt.plot(range(5), result['eigenvalues'][:5], 'o-', label=pattern_name)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Energy')
        plt.title('Energy Eigenvalues for Different K-Coupling Patterns')
        plt.legend()
        plt.grid(True)
        plt.savefig('position_dependent_k_eigenvalues.png', dpi=300)
        plt.show()
        
        # 2. Compare dominant frequencies
        plt.figure(figsize=(12, 8))
        labels = list(results.keys())
        x = np.arange(len(labels))
        width = 0.2
        
        # Get up to 3 dominant frequencies for each pattern
        for i in range(3):
            freqs = []
            for pattern_name in labels:
                if i < len(results[pattern_name]['peak_freqs']):
                    freqs.append(results[pattern_name]['peak_freqs'][i])
                else:
                    freqs.append(0)
            plt.bar(x + i*width, freqs, width, label=f'Peak {i+1}')
        
        plt.xlabel('K-Coupling Pattern')
        plt.ylabel('Frequency (Hz)')
        plt.title('Dominant Frequencies for Different K-Coupling Patterns')
        plt.xticks(x + width, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('position_dependent_k_frequencies.png', dpi=300)
        plt.show()
    
    return results