# quantum_sim/examples/superposition_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from basis import generate_selective_basis, initialize_superposition_rho
from evolution import evolve_system_with_selective_k
from analysis import analyze_fft
from visualization import combined_visualization

def run_superposition_simulation():
    """Run quantum simulation with a superposition initial state"""
    # Simulation parameters
    n_qubits = 6
    dt = 0.01
    t_max = 50
    J_max = 1.0
    E_site = 0.0
    analyze_qubit = 4
    
    print("Running simulation for superposition initial state")
    print(f"On-site energy: {E_site}")
    print(f"J_max: {J_max}")
    
    # Generate basis states
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits,
        top_excitations=True,
        bottom_excitations=True
    )
    
    # Display sample basis states
    print("\nSample basis states:")
    for i, state in enumerate(basis_states[:5]):
        print(f"State {i}: {state}")
    if len(basis_states) > 5:
        print(f"... ({len(basis_states) - 5} more states)")
    
    
 # Define amplitudes for the superposition state    
    a_magnitude = np.abs(1 + 0j)                # = 1.0
    b_magnitude = np.abs(0.5 + 0.5j)            # = 0.7071...
    c_magnitude = np.abs(0.3 - 0.2j)            # = 0.36...
     
    a = a_magnitude * np.exp(1j * np.pi/2)      # Same magnitude as original a, but phase of π/2
    b = b_magnitude * np.exp(1j * np.pi)        # Same magnitude as original b, but phase of π
    c = c_magnitude * np.exp(-1j * np.pi/4)     # Same magnitude as original c, but phase of -π/4
     
    
    superposition_dict = {
        (1, 1, 0, 0, 0, 0): a,  # |1⟩|100⟩
        (1, 0, 0, 1, 0, 0): b,  # |1⟩|010⟩
        (1, 0, 0, 0, 0, 1): c   # |1⟩|001⟩
    }
    
    # Initialize density matrix from superposition
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # For the superposition simulation, use different K values for each top-bottom pair
    k_pattern = {
        (0, 1): 0.5,  # K value for top qubit 0 and bottom qubit 1
        (2, 3): 1.0,  # K value for top qubit 2 and bottom qubit 3
        (4, 5): 1.5   # K value for top qubit 4 and bottom qubit 5
        }
    pattern_name = f"Superposition Bottom with Top[0] Excited, Non-uniform K: (0,1)→{k_pattern[(0,1)]}, (2,3)→{k_pattern[(2,3)]}, (4,5)→{k_pattern[(4,5)]}"
    
    print("\n===== Superposition of Bottom Qubits with Top[0] Excited =====")
    print("Superposition state amplitudes:")
    for state, amp in superposition_dict.items():
        print(f"State {state}: amplitude {amp}")
    
    # Evolve the system
    probabilities_1, times, eigenvalues = evolve_system_with_selective_k(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        J_max=J_max, k_pattern=k_pattern, E_site=E_site, initial_rho=initial_rho
    )
    
    # Perform FFT analysis
    print(f"\n--- Analyzing FFT for qubit {analyze_qubit} (Superposition) ---")
    freqs_pos, abs_fft, peak_freqs, peak_amps = analyze_fft(
        probabilities_1, times, analyze_qubit
    )
    fft_data = (freqs_pos, abs_fft, peak_freqs, peak_amps)
    
    # Create visualization
    fig = combined_visualization(
        eigenvalues=eigenvalues,
        eigenvectors=None,  # We don't have eigenvectors here
        basis_states=basis_states,
        probabilities_1=probabilities_1,
        times=times,
        analyze_qubit=analyze_qubit,
        pattern_name=pattern_name,
        k_pattern=k_pattern,
        initial_qubits=["Superposition"],
        fft_data=fft_data
    )
    
    # Save and display the figure
    k_pattern_str = f"K_0-1_{k_pattern[(0,1)]}_2-3_{k_pattern[(2,3)]}_4-5_{k_pattern[(4,5)]}"
    fig.savefig(f"quantum_superposition_bottom_with_top0_{k_pattern_str}_analyze_q{analyze_qubit}.png", dpi=300)
    plt.show()
    
    # Print dominant frequencies
    print(f"\nDominant frequencies for Superposition (qubit {analyze_qubit}):")
    print("Freq (Hz) | Amplitude")
    print("-" * 25)
    for freq, amp in zip(peak_freqs[:5], peak_amps[:5]):
        print(f"{freq:8.4f} | {amp:10.4f}")

if __name__ == "__main__":
    run_superposition_simulation()