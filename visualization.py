# quantum_sim/visualization/visualization.py
import numpy as np
import matplotlib.pyplot as plt

def combined_visualization(eigenvalues, eigenvectors, basis_states, probabilities_1, times, 
                           analyze_qubit, pattern_name, k_pattern, initial_qubits, fft_data=None):
    """
    Create a combined visualization with the dynamics and FFT plots.
    """
    fig, (ax_dynamics, ax_fft) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Quantum Analysis: {pattern_name}", fontsize=16, fontweight='bold')
    
    n_qubits = probabilities_1.shape[1]
    bottom_qubits = list(range(1, n_qubits, 2))
    
    # Create a label for the bottom configuration
    if isinstance(initial_qubits, list) and "Superposition" not in initial_qubits:
        bottom_config = "".join("1" if q in initial_qubits and q in bottom_qubits else "0" for q in bottom_qubits)
    else:
        bottom_config = "Superposition"
    
    ax_dynamics.set_title(
        f"Qubit Excitation Dynamics (Initial: Top [0] | Bottom: |{bottom_config}>)",
        fontsize=14
    )
    
    # Plot excitation probabilities
    qubit_colors = plt.cm.tab10(np.linspace(0, 1, n_qubits))
    for q in range(n_qubits):
        if q % 2 == 0:
            linestyle = '-'
            qubit_type = "Top"
            if q+1 < n_qubits and (q, q+1) in k_pattern:
                k_val = k_pattern[(q, q+1)]
                label = f"{qubit_type} Qubit {q} (K={k_val:.1f})"
            else:
                label = f"{qubit_type} Qubit {q}"
        else:
            linestyle = '--'
            qubit_type = "Bottom"
            if q-1 >= 0 and (q-1, q) in k_pattern:
                k_val = k_pattern[(q-1, q)]
                label = f"{qubit_type} Qubit {q} (K={k_val:.1f})"
            else:
                label = f"{qubit_type} Qubit {q}"
        
        ax_dynamics.plot(times, probabilities_1[:, q], 
                         label=label,
                         linestyle=linestyle, 
                         color=qubit_colors[q], 
                         linewidth=2)
    
    ax_dynamics.set_xlabel("Time", fontsize=12)
    ax_dynamics.set_ylabel("Excitation Probability", fontsize=12)
    ax_dynamics.grid(True, alpha=0.3)
    ax_dynamics.legend(loc='upper right', ncol=2)
    
    # FFT subplot
    ax_fft.set_title(f"FFT Analysis for Qubit {analyze_qubit}", fontsize=14)
    if fft_data is not None:
        freqs_pos, abs_fft, peak_freqs, peak_amps = fft_data
        ax_fft.plot(freqs_pos, abs_fft, color='navy', linewidth=1.5)
        ax_fft.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_fft.set_ylabel("Amplitude", fontsize=12)
        ax_fft.grid(True, alpha=0.3)
        ax_fft.set_xlim(0, min(2, max(freqs_pos)))
        
        for freq, amp in zip(peak_freqs[:5], peak_amps[:5]):
            ax_fft.plot(freq, amp, 'o', color='red', markersize=8)
            ax_fft.text(freq + 0.05, amp, f"{freq:.4f} Hz", 
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        fontsize=10)
            
    return fig