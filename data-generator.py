#!/usr/bin/env python3
"""
Data Generator for ML Training - Quantum Spin Network with Varying K-Coupling Values
"""

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# First try to import from modular structure
try:
    from basis import generate_selective_basis, initialize_superposition_rho
    from hamiltonian import generate_hamiltonian_with_selective_k
    from evolution import evolve_system_with_selective_k
    from analysis import analyze_fft
    from visualization import combined_visualization
    print("Successfully imported from modular project structure")
except ImportError as e:
    print(f"Modular import failed with error: {e}")
    print("Falling back to monolithic script import")
    
    # Import from monolithic script
    from good10_superposition_commented import (
        generate_selective_basis, initialize_superposition_rho, initialize_restricted_rho,
        generate_hamiltonian_with_selective_k, construct_observable_in_restricted_basis,
        evolve_system_with_selective_k, analyze_fft, combined_visualization
    )

def run_simulation_with_k_values(k01, k23, k45, save_plots=False, plot_dir='plots'):
    """
    Run a quantum spin network simulation with specific K-coupling values and return extracted features.
    
    Parameters:
    -----------
    k01 : float - K-coupling value between qubits 0 and 1
    k23 : float - K-coupling value between qubits 2 and 3
    k45 : float - K-coupling value between qubits 4 and 5
    save_plots : bool - Whether to save visualization plots
    plot_dir : str - Directory to save plots to
    
    Returns:
    --------
    dict : Dictionary containing extracted features from the simulation
    """
    # Simulation parameters
    n_qubits = 6
    dt = 0.01
    t_max = 50
    J_max = 1.0
    E_site = 0.0
    analyze_qubit = 4
    
    print(f"\nRunning simulation with K-couplings: K01={k01}, K23={k23}, K45={k45}")
    
    # Generate basis states
    basis_states, state_to_idx = generate_selective_basis(
        n_qubits,
        top_excitations=True,
        bottom_excitations=True
    )
    
    # Define initial superposition state
    # Define amplitudes with phases
    a_magnitude = 1.0
    b_magnitude = np.sqrt(0.5**2 + 0.5**2)  # ≈ 0.7071
    c_magnitude = np.sqrt(0.3**2 + 0.2**2)  # ≈ 0.36
     
    a = a_magnitude * np.exp(1j * np.pi/2)      # Same magnitude with phase π/2
    b = b_magnitude * np.exp(1j * np.pi)        # Same magnitude with phase π
    c = c_magnitude * np.exp(-1j * np.pi/4)     # Same magnitude with phase -π/4
    
    superposition_dict = {
        (1, 1, 0, 0, 0, 0): a,  # |1⟩|100⟩
        (1, 0, 0, 1, 0, 0): b,  # |1⟩|010⟩
        (1, 0, 0, 0, 0, 1): c   # |1⟩|001⟩
    }
    
    # Initialize density matrix from superposition
    initial_rho = initialize_superposition_rho(basis_states, state_to_idx, n_qubits, superposition_dict)
    
    # Define K-coupling pattern with provided values
    k_pattern = {
        (0, 1): k01,
        (2, 3): k23,
        (4, 5): k45
    }
    
    pattern_name = f"K-Coupling: (0,1)→{k01}, (2,3)→{k23}, (4,5)→{k45}"
    
    # Generate Hamiltonian
    H, eigenvalues = generate_hamiltonian_with_selective_k(
        basis_states, state_to_idx, n_qubits, J_max, k_pattern, E_site
    )
    
    # Evolve the system
    start_time = time.time()
    probabilities_1, times, _ = evolve_system_with_selective_k(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        J_max=J_max, k_pattern=k_pattern, E_site=E_site, initial_rho=initial_rho
    )
    sim_time = time.time() - start_time
    print(f"Simulation completed in {sim_time:.2f} seconds")
    
    # Perform FFT analysis for the specified qubit
    freqs_pos, abs_fft, peak_freqs, peak_amps = analyze_fft(
        probabilities_1, times, analyze_qubit
    )
    
    # Feature extraction
    features = {}
    
    # 1. Input parameters
    features['k01'] = k01
    features['k23'] = k23
    features['k45'] = k45
    
    # 2. Top eigenvalues (first 8)
    for i in range(min(8, len(eigenvalues))):
        features[f'eigenvalue_{i}'] = eigenvalues[i]
    
    # 3. FFT peak frequencies and amplitudes (up to 5 peaks)
    for i in range(min(5, len(peak_freqs))):
        if i < len(peak_freqs):
            features[f'peak_freq_{i}'] = peak_freqs[i]
            features[f'peak_amp_{i}'] = peak_amps[i]
        else:
            features[f'peak_freq_{i}'] = 0.0
            features[f'peak_amp_{i}'] = 0.0
    
    # Save plots if requested
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        
        fft_data = (freqs_pos, abs_fft, peak_freqs, peak_amps)
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
        
        # Create filename using K values
        filename = f"k01_{k01}_k23_{k23}_k45_{k45}.png"
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=200)
        plt.close(fig)
        
        print(f"Visualization saved to {filepath}")
    
    return features

def generate_training_data(num_samples=125, k_min=0.1, k_max=3.0, save_plots=False, save_interval=10):
    """
    Generate a dataset by running simulations with random K-coupling configurations.
    
    Parameters:
    -----------
    num_samples : int - Number of random configurations to generate
    k_min : float - Minimum K-coupling value
    k_max : float - Maximum K-coupling value
    save_plots : bool - Whether to save visualization plots
    save_interval : int - Save CSV file every N simulations
    
    Returns:
    --------
    pandas.DataFrame : DataFrame containing all simulation results
    """
    print(f"Generating training data with {num_samples} random K-coupling configurations")
    print(f"K-value range: [{k_min}, {k_max}]")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create plot directory if saving plots
    plot_dir = f'plots_{timestamp}'
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize list to store results
    all_results = []
    
    # Generate random K-coupling configurations
    np.random.seed(42)  # For reproducibility
    
    # Run simulations for each configuration
    for i in range(num_samples):
        config_id = i + 1
        
        # Generate random K values for this configuration
        k01 = round(np.random.uniform(k_min, k_max), 2)
        k23 = round(np.random.uniform(k_min, k_max), 2)
        k45 = round(np.random.uniform(k_min, k_max), 2)
        
        print(f"\nConfiguration {config_id}/{num_samples}")
        print(f"Random K values: k01={k01}, k23={k23}, k45={k45}")
        
        # Run simulation and extract features
        features = run_simulation_with_k_values(k01, k23, k45, save_plots, plot_dir)
        
        # Add configuration ID
        features['config_id'] = config_id
        
        # Add to results list
        all_results.append(features)
        
        # Save intermediate results periodically
        if i % save_interval == 0 or i == num_samples - 1:
            # Convert to DataFrame
            df = pd.DataFrame(all_results)
            
            # Save to CSV
            csv_filename = f'quantum_spin_data_{timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Saved intermediate results to {csv_filename} ({len(df)} configurations)")
    
    # Final conversion to DataFrame
    df_final = pd.DataFrame(all_results)
    
    # Save final results
    final_csv_filename = f'quantum_spin_data_{timestamp}_final.csv'
    df_final.to_csv(final_csv_filename, index=False)
    print(f"\nCompleted data generation. Final dataset saved to {final_csv_filename}")
    print(f"Total configurations: {len(df_final)}")
    
    return df_final

def main():
    """Main function to run the data generation"""
    print("Quantum Spin Network - Data Generation for ML Training")
    print("====================================================")
    
    # Ask for number of samples
    num_samples_str = input("Enter number of random configurations to generate (recommended: 100-500): ")
    num_samples = int(num_samples_str) if num_samples_str.isdigit() else 125
    
    # Ask for K value range
    k_min_str = input("Enter minimum K value (default: 0.1): ")
    k_min = float(k_min_str) if k_min_str and k_min_str.replace('.', '', 1).isdigit() else 0.1
    
    k_max_str = input("Enter maximum K value (default: 3.0): ")
    k_max = float(k_max_str) if k_max_str and k_max_str.replace('.', '', 1).isdigit() else 3.0
    
    # Confirm with user
    print(f"\nThis will generate data for {num_samples} random K-coupling configurations.")
    print(f"K value range: [{k_min}, {k_max}]")
    
    proceed = input("\nProceed with data generation? (y/n): ")
    
    if proceed.lower() == 'y':
        # Ask if plots should be saved
        save_plots = input("Save plots for each configuration? (y/n): ").lower() == 'y'
        
        # Set save interval based on total configurations
        save_interval = max(1, num_samples // 10)
        
        # Generate the data
        df = generate_training_data(num_samples, k_min, k_max, save_plots, save_interval)
        
        print("\nData generation complete!")
        print(f"Generated {len(df)} configurations with {df.shape[1]} features per configuration.")
    else:
        print("Data generation cancelled.")

if __name__ == "__main__":
    main()