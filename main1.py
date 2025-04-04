#!/usr/bin/env python3
"""
Quantum Spin Network Simulation
Main entry point for running different simulations
"""
import os
import sys
import importlib
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Make sure output directory exists
os.makedirs('output', exist_ok=True)

def run_module(module_name):
    """Run a module by importing and executing its main function"""
    try:
        module = importlib.import_module(module_name)
        
        # For modules with self-contained main functions
        
        # Add run_superposition_simulation to the checks
        if hasattr(module, 'run_superposition_simulation') and callable(module.run_superposition_simulation):
            module.run_superposition_simulation()
        elif hasattr(module, 'run_simulation') and callable(module.run_simulation):
            module.run_simulation()
        if hasattr(module, 'run_simulation') and callable(module.run_simulation):
            module.run_simulation()
        elif hasattr(module, 'run_modified_quantum_simulation') and callable(module.run_modified_quantum_simulation):
            module.run_modified_quantum_simulation()
        elif hasattr(module, 'analyze_k_coupling_effects') and callable(module.analyze_k_coupling_effects):
            module.analyze_k_coupling_effects()
        elif hasattr(module, 'compare_classical_superposition') and callable(module.compare_classical_superposition):
            module.compare_classical_superposition()
        elif hasattr(module, 'analyze_reduced_density_matrix') and callable(module.analyze_reduced_density_matrix):
            # Run partial trace analysis with default parameters
            run_partial_trace_analysis()
        
        # Special case for fidelity.py which seems to need parameters
        elif module_name == "fidelity" and hasattr(module, 'analyze_fidelity'):
            # Import necessary modules for fidelity analysis
            from basis import generate_selective_basis, initialize_superposition_rho
            from hamiltonian import generate_hamiltonian_with_selective_k
            
            # Set up parameters
            n_qubits = 6
            dt = 0.01
            t_max = 50
            
            # Generate basis states
            basis_states, state_to_idx = generate_selective_basis(
                n_qubits, 
                top_excitations=True, 
                bottom_excitations=True
            )
            
            # Define the superposition state
            superposition_dict = {
                (1, 1, 0, 0, 0, 0): 1.0 + 0j,
                (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,
                (1, 0, 0, 0, 0, 1): 0.3 - 0.2j
            }
            
            # Generate Hamiltonian
            k_pattern = {(0, 1): 1.0, (2, 3): 1.0, (4, 5): 1.0}
            H, _ = generate_hamiltonian_with_selective_k(
                basis_states, state_to_idx, n_qubits, 
                J_max=1.0, k_pattern=k_pattern, E_site=0.0
            )
            
            # Call analyze_fidelity with needed parameters
            module.analyze_fidelity(
                basis_states=basis_states,
                state_to_idx=state_to_idx,
                n_qubits=n_qubits,
                dt=dt,
                t_max=t_max,
                superposition_dict=superposition_dict,
                H=H
            )
        else:
            print(f"Module {module_name} has no runnable function found")
            print(f"Available functions: {[f for f in dir(module) if callable(getattr(module, f)) and not f.startswith('_')]}")
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        print(f"Make sure the file {module_name}.py exists in the current directory.")
    except Exception as e:
        print(f"Error running {module_name}: {e}")

def run_partial_trace_analysis(use_nonuniform_k=False):
    """Run analysis of partial trace for the quantum spin network"""
    print("\n=== Running Partial Trace Analysis ===")
    print("Checking if Tr0,2,4[rho(t)]=rho(t)1,3,5 throughout time evolution")
    
    try:
        # Import necessary modules
        from basis import generate_selective_basis, initialize_superposition_rho
        from hamiltonian import generate_hamiltonian_with_selective_k
        from quantum_ops import rk4_step
        from partial_trace import (
            partial_trace, 
            analyze_reduced_density_matrix,
            analyze_entanglement_over_time
        )
        
        # Set up parameters
        n_qubits = 6
        dt = 0.01
        t_max = 50
        
        # Generate basis states
        basis_states, state_to_idx = generate_selective_basis(
            n_qubits, 
            top_excitations=True, 
            bottom_excitations=True
        )
        
        # Define the superposition state
        superposition_dict = {
            (1, 1, 0, 0, 0, 0): 1.0 + 0j,
            (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,
            (1, 0, 0, 0, 0, 1): 0.3 - 0.2j
        }
        
        # Initialize density matrix
        initial_rho = initialize_superposition_rho(
            basis_states, state_to_idx, n_qubits, superposition_dict
        )
        
        # Define K coupling pattern
        if use_nonuniform_k:
            k_pattern = {
                (0, 1): 1.5,
                (2, 3): 1.0,
                (4, 5): 0.5
            }
            pattern_name = "Non-uniform K coupling"
        else:
            k_pattern = {
                (0, 1): 1.0,
                (2, 3): 1.0,
                (4, 5): 1.0
            }
            pattern_name = "Uniform K=1.0 coupling"
        
        print(f"Using {pattern_name}")
        
        # Generate Hamiltonian
        H, eigenvalues = generate_hamiltonian_with_selective_k(
            basis_states, state_to_idx, n_qubits, 
            J_max=1.0, k_pattern=k_pattern, E_site=0.0
        )
        
        # Analyze initial reduced density matrix
        print("\nInitial reduced density matrix (t=0):")
        reduced_rho_initial, state_map = partial_trace(
            initial_rho, basis_states, [0, 2, 4], n_qubits
        )
        print(np.round(reduced_rho_initial, 3))
        
        # Verify trace = 1
        trace_value = np.real(np.trace(reduced_rho_initial))
        print(f"Trace of reduced density matrix: {trace_value:.6f} (should be 1.0)")
        
        # Show probabilities for different states of bottom qubits
        diag_elements = np.real(np.diag(reduced_rho_initial))
        print("\nBottom qubit state probabilities at t=0:")
        for idx, prob in enumerate(diag_elements):
            if prob > 0.01:  # Only show non-negligible probabilities
                print(f"State |{state_map[idx]}⟩: {prob:.4f}")
        
        # Run detailed analysis of reduced density matrix over time
        print("\nAnalyzing reduced density matrix at multiple time points...")
        analyze_reduced_density_matrix(
            basis_states, state_to_idx, n_qubits, dt, t_max,
            initial_rho, H, 
            qubits_to_trace_out=[0, 2, 4],  # Trace out top qubits
            sample_times=[0, 12.5, 25, 37.5, 50]  # Sample at these times
        )
        
        # Analyze entanglement over time
        print("\nAnalyzing entanglement measures over time...")
        analyze_entanglement_over_time(
            basis_states, state_to_idx, n_qubits, dt, t_max,
            initial_rho, H
        )
        
        print("\nPartial trace analysis complete!")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure partial_trace.py and other required files exist in the current directory.")
    except Exception as e:
        print(f"Error during partial trace analysis: {e}")

def main():
    """Main function to run the quantum spin network simulations"""
    print("Quantum Spin Network Simulation")
    print("===============================")
    
    print("\nSelect a simulation to run:")
    print("1. Basic superposition simulation")
    print("2. Compare classical vs superposition states")
    print("3. Analyze K-coupling effects")
    print("4. Run fidelity analysis")
    print("5. Run partial trace analysis (uniform K)")
    print("6. Run partial trace analysis (non-uniform K)")
    print("7. Run all simulations")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-7): ")
    
    if choice == '1':
        run_module("superposition_simulation")
    elif choice == '2':
        run_module("classical_vs_superposition")
    elif choice == '3':
        run_module("analyze_k_coupling_effects")
    elif choice == '4':
        run_module("fidelity")
    elif choice == '5':
        run_partial_trace_analysis(use_nonuniform_k=False)
    elif choice == '6':
        run_partial_trace_analysis(use_nonuniform_k=True)
    elif choice == '7':
        run_module("superposition_simulation")
        run_module("classical_vs_superposition")
        run_module("analyze_k_coupling_effects")
        run_module("fidelity")
        run_partial_trace_analysis(use_nonuniform_k=False)
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()