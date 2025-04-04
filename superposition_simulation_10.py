# quantum_sim/examples/superposition_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from basis import generate_selective_basis, initialize_superposition_rho
from evolution import evolve_system_with_selective_k
from analysis import analyze_fft
from visualization import combined_visualization
import time
import datetime

def format_time(seconds):
    """Format seconds into a readable time string"""
    return str(datetime.timedelta(seconds=int(seconds)))

def run_superposition_simulation():
    """Run quantum simulation with a superposition initial state for 10 qubits"""
    # Start overall timing
    total_start_time = time.time()
    
    # Simulation parameters
    n_qubits = 10  # Modified from 6 to 10 qubits
    dt = 0.01
    t_max = 50
    J_max = 1.0
    E_site = 0.0
    analyze_qubit = 8  # Changed from 4 to 6 as we have more qubits now
    
    print("Running simulation for superposition initial state with 10 qubits")
    print(f"On-site energy: {E_site}")
    print(f"J_max: {J_max}")
    
    # Generate basis states
    print("\nGenerating basis states...")
    basis_start_time = time.time()
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits,
        top_excitations=True,
        bottom_excitations=True
    )
    basis_time = time.time() - basis_start_time
    print(f"Basis generation completed in {basis_time:.2f} seconds")
    print(f"Hilbert space dimension: {len(basis_states)}")
    
    # Display sample basis states
    print("\nSample basis states:")
    for i, state in enumerate(basis_states[:5]):
        print(f"State {i}: {state}")
    if len(basis_states) > 5:
        print(f"... ({len(basis_states) - 5} more states)")
    
    # Identify top and bottom qubits
    top_qubits = list(range(0, n_qubits, 2))  # Even indices (0,2,4,6,8)
    bottom_qubits = list(range(1, n_qubits, 2))  # Odd indices (1,3,5,7,9)
    
    # Define amplitudes for the superposition state with phases
    print("\nDefining superposition state...")
    a = 1.0 * np.exp(1j * np.pi/4)  # Phase of π/4
    b = 0.8 * np.exp(1j * np.pi/2)  # Phase of π/2
    c = 0.6 * np.exp(1j * np.pi)    # Phase of π
    d = 0.4 * np.exp(-1j * np.pi/3) # Phase of -π/3
    e = 0.2 * np.exp(-1j * np.pi/6) # Phase of -π/6
    
    # Create a superposition state with top qubit 0 always excited
    # and one of the bottom qubits excited with different amplitudes
    superposition_dict = {
        # Format: (top qubits even indices, bottom qubits odd indices)
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0): a,  # |1⟩|10000⟩ - qubit 0 and 1 excited
        (1, 0, 0, 1, 0, 0, 0, 0, 0, 0): b,  # |1⟩|01000⟩ - qubit 0 and 3 excited
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0): c,  # |1⟩|00100⟩ - qubit 0 and 5 excited
        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0): d,  # |1⟩|00010⟩ - qubit 0 and 7 excited
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 1): e,  # |1⟩|00001⟩ - qubit 0 and 9 excited
    }
    
    # Initialize density matrix from superposition
    print("Initializing density matrix...")
    init_start_time = time.time()
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    init_time = time.time() - init_start_time
    print(f"Density matrix initialization completed in {init_time:.2f} seconds")
    
    # Create an interesting pattern of K couplings
    print("Setting up K-coupling pattern...")
    # Using a gradient pattern where coupling strength increases along the chain
    k_pattern = {}
    for i, (t, b) in enumerate(zip(top_qubits, bottom_qubits)):
        # Create a gradient of K values from 0.5 to 2.0
        k_pattern[(t, b)] = 0.5 + (1.5 * i / (len(top_qubits) - 1))
    
    pattern_description = "Gradient K-coupling from 0.5 to 2.0"
    pattern_name = f"10-Qubit Superposition with {pattern_description}"
    
    print("\n===== 10-Qubit Superposition with Top[0] Excited =====")
    print("Superposition state amplitudes:")
    for state, amp in superposition_dict.items():
        state_str = ''.join(map(str, state))
        amp_abs = abs(amp)
        amp_phase = np.angle(amp)
        print(f"State |{state_str}⟩: amplitude {amp} (magnitude: {amp_abs:.2f}, phase: {amp_phase:.2f} rad)")
    
    print("\nK-coupling pattern:")
    for (t, b), k in k_pattern.items():
        print(f"K({t},{b}) = {k:.2f}")
    
    # Evolve the system
    print("\nStarting time evolution...")
    print(f"Time steps: {int(t_max/dt) + 1}, dt = {dt}, t_max = {t_max}")
    evolution_start_time = time.time()
    prob_update_interval = max(1, int((t_max/dt) / 10))  # Update progress every ~10%
    
    # Patch the evolve_system_with_selective_k function to report progress
    # Since we can't modify it directly, we'll use a wrapper to monitor progress
    def evolve_with_progress_tracking():
        # First, generate the Hamiltonian (can be time-consuming for large systems)
        print("Generating Hamiltonian...")
        hamiltonian_start = time.time()
        from hamiltonian import generate_hamiltonian_with_selective_k as gen_hamiltonian
        H, eigenvalues = gen_hamiltonian(
            basis_states, state_to_idx, n_qubits, J_max, k_pattern, E_site
        )
        hamiltonian_time = time.time() - hamiltonian_start
        print(f"Hamiltonian generation completed in {hamiltonian_time:.2f} seconds")
        
        # Create projection operators
        print("Creating projection operators...")
        projector_start = time.time()
        from observables import construct_observable_in_restricted_basis
        projectors = [
            construct_observable_in_restricted_basis(basis_states, state_to_idx, n_qubits, q, 'projector_1') 
            for q in range(n_qubits)
        ]
        projector_time = time.time() - projector_start
        print(f"Projector creation completed in {projector_time:.2f} seconds")
        
        # Set up time points and array to store results
        tpoints = np.arange(0, t_max + dt, dt)
        n_steps = len(tpoints)
        probabilities_1 = np.zeros((n_steps, n_qubits))
        
        # Initial probabilities
        for q in range(n_qubits):
            probabilities_1[0, q] = np.real(np.trace(initial_rho @ projectors[q]))
        
        # Time evolution with progress reporting
        print("Evolving system...")
        rho = initial_rho.copy()
        evolution_start = time.time()
        from quantum_ops import rk4_step
        
        for t_idx in range(1, n_steps):
            # Evolve one step
            rho = rk4_step(H, rho, dt)
            
            # Calculate probabilities
            for q in range(n_qubits):
                probabilities_1[t_idx, q] = np.real(np.trace(rho @ projectors[q]))
            
            # Report progress
            if t_idx % prob_update_interval == 0 or t_idx == n_steps - 1:
                progress = (t_idx / (n_steps - 1)) * 100
                elapsed = time.time() - evolution_start
                estimated_total = elapsed / (t_idx / (n_steps - 1))
                remaining = estimated_total - elapsed
                print(f"Evolution progress: {progress:.1f}% - Step {t_idx}/{n_steps-1} - "
                      f"Elapsed: {format_time(elapsed)}, Remaining: {format_time(remaining)}")
        
        evolution_time = time.time() - evolution_start
        print(f"Time evolution completed in {format_time(evolution_time)}")
        steps_per_second = (n_steps - 1) / evolution_time
        print(f"Performance: {steps_per_second:.2f} time steps per second")
        
        return probabilities_1, tpoints, eigenvalues
    
    # Run the evolution with progress tracking
    probabilities_1, times, eigenvalues = evolve_with_progress_tracking()
    
    evolution_time = time.time() - evolution_start_time
    print(f"Total evolution process completed in {format_time(evolution_time)}")
    
    # Perform FFT analysis
    print(f"\n--- Analyzing FFT for qubit {analyze_qubit} ---")
    fft_start_time = time.time()
    freqs_pos, abs_fft, peak_freqs, peak_amps = analyze_fft(
        probabilities_1, times, analyze_qubit
    )
    fft_time = time.time() - fft_start_time
    print(f"FFT analysis completed in {fft_time:.2f} seconds")
    
    fft_data = (freqs_pos, abs_fft, peak_freqs, peak_amps)
    
    # Create visualization
    print("\nGenerating visualization...")
    vis_start_time = time.time()
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
    fig.savefig(f"10qubit_superposition_gradient_k_analyze_q{analyze_qubit}.png", dpi=300)
    plt.show()
    vis_time = time.time() - vis_start_time
    print(f"Visualization completed in {vis_time:.2f} seconds")
    
    # Print dominant frequencies
    print(f"\nDominant frequencies for qubit {analyze_qubit}:")
    print("Freq (Hz) | Amplitude")
    print("-" * 25)
    for freq, amp in zip(peak_freqs[:5], peak_amps[:5]):
        print(f"{freq:8.4f} | {amp:10.4f}")
    
    # Calculate time-averaged excitation probabilities for each qubit
    avg_probs = np.mean(probabilities_1, axis=0)
    
    # Print time-averaged excitation probabilities
    print("\nTime-averaged excitation probabilities:")
    for q in range(n_qubits):
        qubit_type = "Top" if q % 2 == 0 else "Bottom"
        print(f"{qubit_type} Qubit {q}: {avg_probs[q]:.4f}")
    
    # Create a bar chart of time-averaged excitation probabilities
    plt.figure(figsize=(12, 6))
    bar_colors = ['blue' if i % 2 == 0 else 'red' for i in range(n_qubits)]
    plt.bar(range(n_qubits), avg_probs, color=bar_colors)
    plt.xlabel('Qubit Index')
    plt.ylabel('Time-Averaged Excitation Probability')
    plt.title('Time-Averaged Excitation Probabilities for 10-Qubit System')
    plt.xticks(range(n_qubits))
    
    # Add "Top" and "Bottom" labels to the x-axis
    for i in range(n_qubits):
        label = "Top" if i % 2 == 0 else "Bottom"
        plt.annotate(f"{label}", 
                    xy=(i, -0.01), 
                    xytext=(0, -10),
                    textcoords='offset points',
                    ha='center', 
                    va='top', 
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig("10qubit_time_averaged_probabilities.png", dpi=300)
    plt.show()
    
    # Print overall timing summary
    total_time = time.time() - total_start_time
    print("\n===== Timing Summary =====")
    print(f"Total simulation time: {format_time(total_time)}")
    print(f"Basis generation: {basis_time:.2f} seconds ({basis_time/total_time*100:.1f}%)")
    print(f"Density matrix init: {init_time:.2f} seconds ({init_time/total_time*100:.1f}%)")
    print(f"Evolution process: {evolution_time:.2f} seconds ({evolution_time/total_time*100:.1f}%)")
    print(f"FFT analysis: {fft_time:.2f} seconds ({fft_time/total_time*100:.1f}%)")
    print(f"Visualization: {vis_time:.2f} seconds ({vis_time/total_time*100:.1f}%)")
    
    # Return the timing information in case needed elsewhere
    timing_info = {
        'total': total_time,
        'basis': basis_time,
        'init': init_time,
        'evolution': evolution_time,
        'fft': fft_time,
        'visualization': vis_time
    }
    
    return timing_info

if __name__ == "__main__":
    run_superposition_simulation()