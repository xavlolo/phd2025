import numpy as np
import matplotlib.pyplot as plt
from basis import generate_selective_basis, initialize_superposition_rho
from hamiltonian import generate_hamiltonian_with_selective_k
from evolution import evolve_system_with_selective_k
from analysis import analyze_fft

# If the imports above fail, try this instead:
# from good10_superposition_commented import (
#     generate_selective_basis, initialize_superposition_rho,
#     generate_hamiltonian_with_selective_k, evolve_system_with_selective_k,
#     analyze_fft
# )

def analyze_k_coupling_effects(k_values=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], plot=True):
    """
    Analyze the effects of different K-coupling strengths on quantum spin network dynamics.
    
    Parameters:
    -----------
    k_values : list, optional
        List of K-coupling strengths to investigate. Default is [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    plot : bool, optional
        Whether to generate visualization. Default is True
    
    Returns:
    --------
    dict : A dictionary containing analysis results for each K value
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

    # Plotting preparation if needed
    if plot:
        fig, axs = plt.subplots(len(k_values), 2, figsize=(16, 4*len(k_values)))
        fig.suptitle("K-Coupling Strength Effects on Quantum Spin Network", fontsize=16)

    # Iterate through different K values
    for k_idx, k_value in enumerate(k_values):
        # Create K-coupling pattern
        k_pattern = {
            (0, 1): k_value,
            (2, 3): k_value,
            (4, 5): k_value
        }

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
        results[k_value] = {
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
            # Dynamics plot
            axs[k_idx, 0].set_title(f'Qubit Dynamics (K = {k_value})')
            for q in range(n_qubits):
                linestyle = '-' if q % 2 == 0 else '--'
                qubit_type = "Top" if q % 2 == 0 else "Bottom"
                axs[k_idx, 0].plot(times, probabilities_1[:, q], 
                                   label=f'{qubit_type} Qubit {q}', 
                                   linestyle=linestyle)
            axs[k_idx, 0].set_xlabel('Time')
            axs[k_idx, 0].set_ylabel('Excitation Probability')
            axs[k_idx, 0].legend(loc='best')
            axs[k_idx, 0].grid(True, alpha=0.3)

            # FFT plot
            axs[k_idx, 1].set_title(f'FFT Analysis for Qubit {analyze_qubit} (K = {k_value})')
            axs[k_idx, 1].plot(freqs_pos, abs_fft, color='navy', linewidth=1.5)
            axs[k_idx, 1].set_xlabel('Frequency (Hz)')
            axs[k_idx, 1].set_ylabel('Amplitude')
            axs[k_idx, 1].grid(True, alpha=0.3)
            axs[k_idx, 1].set_xlim(0, min(2, max(freqs_pos)))

            # Mark peaks
            for freq, amp in zip(peak_freqs[:3], peak_amps[:3]):
                axs[k_idx, 1].plot(freq, amp, 'ro', markersize=6)
                axs[k_idx, 1].text(freq + 0.02, amp, f'{freq:.4f} Hz', 
                                   verticalalignment='center',
                                   fontsize=8)

    # Finalize plot
    if plot:
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('k_coupling_effects.png', dpi=300)
        plt.show()

    # Additional summary analysis - create comparative plots of frequencies and eigenvalues
    if plot and len(k_values) > 1:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Plot energy eigenvalues vs K value
        for k_idx, k_value in enumerate(k_values):
            # Get first 5 eigenvalues for each K
            eigs = results[k_value]['eigenvalues'][:5]
            axs[0, 0].plot(np.ones(len(eigs))*k_value, eigs, 'o-', label=f'K={k_value}')
        
        axs[0, 0].set_xlabel('K Value')
        axs[0, 0].set_ylabel('Energy Eigenvalues')
        axs[0, 0].set_title('Energy Eigenvalues vs K-Coupling Strength')
        axs[0, 0].grid(True)
        
        # 2. Plot energy gaps between consecutive levels vs K value
        for k_idx, k_value in enumerate(k_values):
            eigs = results[k_value]['eigenvalues'][:5]
            gaps = np.diff(eigs)
            axs[0, 1].plot(range(len(gaps)), gaps, 'o-', label=f'K={k_value}')
        
        axs[0, 1].set_xlabel('Gap Index')
        axs[0, 1].set_ylabel('Energy Gap')
        axs[0, 1].set_title('Energy Gaps vs K-Coupling Strength')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 3. Plot first 3 FFT peak frequencies for each K value
        peak_freqs_data = []
        for k_value in k_values:
            freqs = results[k_value]['peak_freqs'][:3]
            # Pad with zeros if fewer than 3 peaks
            freqs = np.pad(freqs, (0, max(0, 3-len(freqs))), mode='constant')
            peak_freqs_data.append(freqs)
        
        peak_freqs_data = np.array(peak_freqs_data)
        for i in range(3):
            axs[1, 0].plot(k_values, peak_freqs_data[:, i], 'o-', label=f'Peak {i+1}')
        
        axs[1, 0].set_xlabel('K Value')
        axs[1, 0].set_ylabel('Frequency (Hz)')
        axs[1, 0].set_title('FFT Peak Frequencies vs K-Coupling Strength')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # 4. Plot FFT amplitudes for each K value
        peak_amps_data = []
        for k_value in k_values:
            amps = results[k_value]['peak_amps'][:3]
            # Pad with zeros if fewer than 3 peaks
            amps = np.pad(amps, (0, max(0, 3-len(amps))), mode='constant')
            peak_amps_data.append(amps)
        
        peak_amps_data = np.array(peak_amps_data)
        for i in range(3):
            axs[1, 1].plot(k_values, peak_amps_data[:, i], 'o-', label=f'Peak {i+1}')
        
        axs[1, 1].set_xlabel('K Value')
        axs[1, 1].set_ylabel('Amplitude')
        axs[1, 1].set_title('FFT Peak Amplitudes vs K-Coupling Strength')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('k_coupling_effects_summary.png', dpi=300)
        plt.show()

    return results