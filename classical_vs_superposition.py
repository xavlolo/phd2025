#!/usr/bin/env python3
"""
Quantum Spin Network Simulation
Compare classical and superposition initial states
"""
import numpy as np
import matplotlib.pyplot as plt

# Try to import from modular structure first
try:
    from basis import generate_selective_basis, initialize_restricted_rho, initialize_superposition_rho
    from hamiltonian import generate_hamiltonian_with_selective_k
    from evolution import evolve_system_with_selective_k
    from analysis import analyze_fft
    print("Successfully imported from modular project structure")
    
# Fall back to monolithic script if modular imports fail
except ImportError as e:
    print(f"Modular import failed with error: {e}")
    print("Falling back to monolithic script import")
    
    # Import everything from the monolithic script
    from good10_superposition_commented import (
        commutator, rk4_step, generate_selective_basis, 
        initialize_restricted_rho, initialize_superposition_rho,
        generate_hamiltonian_with_selective_k, construct_observable_in_restricted_basis,
        evolve_system_with_selective_k, analyze_fft
    )

def compare_classical_superposition():
    """Compare classical and superposition initial states"""
    # Simulation parameters
    n_qubits = 6
    dt = 0.01
    t_max = 50
    J_max = 1.0
    E_site = 0.0
    analyze_qubit = 4
    
    print("\n=== Comparing Classical vs Superposition States ===")
    
    # Generate basis states
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits,
        top_excitations=True,
        bottom_excitations=True
    )
    
    # Set up K coupling pattern (uniform)
    top_qubits = list(range(0, n_qubits, 2))
    bottom_qubits = list(range(1, n_qubits, 2))
    k_value = 1.0
    k_pattern = {(t, b): k_value for t, b in zip(top_qubits, bottom_qubits)}
    
    # Run simulation with classical initial state (only qubit 0 excited)
    print("\n--- Classical Initial State |100000⟩ ---")
    classical_probs, classical_times, _ = evolve_system_with_selective_k(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        qubits_to_excite=[0],  # Only top qubit 0 excited
        J_max=J_max, k_pattern=k_pattern, E_site=E_site
    )
    
    # Run FFT analysis
    classical_fft = analyze_fft(classical_probs, classical_times, analyze_qubit)
    
    # Run simulation with superposition initial state
    print("\n--- Superposition Initial State ---")
    # Define the superposition
    superposition_dict = {
        (1, 1, 0, 0, 0, 0): 1.0 + 0j,      # |110000⟩
        (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,    # |100100⟩
        (1, 0, 0, 0, 0, 1): 0.3 - 0.2j      # |100001⟩
    }
    
    # Initialize density matrix from superposition
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # Evolve the system
    super_probs, super_times, _ = evolve_system_with_selective_k(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        J_max=J_max, k_pattern=k_pattern, E_site=E_site, initial_rho=initial_rho
    )
    
    # Run FFT analysis
    super_fft = analyze_fft(super_probs, super_times, analyze_qubit)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classical vs Superposition Initial States", fontsize=16)
    
    # Dynamics plots
    axes[0, 0].set_title("Classical Initial State - Dynamics")
    axes[0, 1].set_title("Superposition Initial State - Dynamics")
    
    # Use same colors for each qubit across both plots
    colors = plt.cm.tab10(np.linspace(0, 1, n_qubits))
    
    for q in range(n_qubits):
        linestyle = '-' if q % 2 == 0 else '--'
        qubit_type = "Top" if q % 2 == 0 else "Bottom"
        
        # Plot classical
        axes[0, 0].plot(classical_times, classical_probs[:, q], 
                     label=f"{qubit_type} Qubit {q}",
                     linestyle=linestyle, 
                     color=colors[q])
        
        # Plot superposition
        axes[0, 1].plot(super_times, super_probs[:, q], 
                     label=f"{qubit_type} Qubit {q}",
                     linestyle=linestyle, 
                     color=colors[q])
    
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Excitation Probability")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Excitation Probability")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.3)
    
    # FFT plots
    axes[1, 0].set_title(f"Classical Initial State - FFT (Qubit {analyze_qubit})")
    axes[1, 1].set_title(f"Superposition Initial State - FFT (Qubit {analyze_qubit})")
    
    freqs_classical, amps_classical, peak_freqs_classical, peak_amps_classical = classical_fft
    freqs_super, amps_super, peak_freqs_super, peak_amps_super = super_fft
    
    # Plot classical FFT
    axes[1, 0].plot(freqs_classical, amps_classical, color='navy')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, min(2, max(freqs_classical)))
    
    # Mark peaks for classical
    for freq, amp in zip(peak_freqs_classical[:5], peak_amps_classical[:5]):
        axes[1, 0].plot(freq, amp, 'ro', markersize=6)
        axes[1, 0].text(freq + 0.02, amp, f"{freq:.4f} Hz", 
                      verticalalignment='center',
                      fontsize=8)
    
    # Plot superposition FFT
    axes[1, 1].plot(freqs_super, amps_super, color='navy')
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, min(2, max(freqs_super)))
    
    # Mark peaks for superposition
    for freq, amp in zip(peak_freqs_super[:5], peak_amps_super[:5]):
        axes[1, 1].plot(freq, amp, 'ro', markersize=6)
        axes[1, 1].text(freq + 0.02, amp, f"{freq:.4f} Hz", 
                      verticalalignment='center',
                      fontsize=8)
    
    plt.tight_layout()
    try:
        plt.savefig('output/classical_vs_superposition.png', dpi=300)
    except Exception as e:
        print(f"Warning: Could not save figure: {e}")
    plt.show()
    
    # Print frequency comparison
    print("\nDominant frequencies comparison:")
    print("\nClassical Initial State:")
    print("Freq (Hz) | Amplitude")
    print("-" * 25)
    for freq, amp in zip(peak_freqs_classical[:5], peak_amps_classical[:5]):
        print(f"{freq:8.4f} | {amp:10.4f}")
    
    print("\nSuperposition Initial State:")
    print("Freq (Hz) | Amplitude")
    print("-" * 25)
    for freq, amp in zip(peak_freqs_super[:5], peak_amps_super[:5]):
        print(f"{freq:8.4f} | {amp:10.4f}")

if __name__ == "__main__":
    compare_classical_superposition()