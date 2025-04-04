# quantum_sim/analysis/analysis.py
import numpy as np

def analyze_fft(probabilities_1, tpoints, qubit_index):
    """Compute FFT and identify peaks"""
    dt = tpoints[1] - tpoints[0]
    prob_signal = probabilities_1[:, qubit_index]
    signal_centered = prob_signal - np.mean(prob_signal)
    window = np.hanning(len(signal_centered))
    windowed_signal = signal_centered * window
    fft_signal = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_signal), dt)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    fft_signal_pos = fft_signal[mask]
    abs_fft = np.abs(fft_signal_pos)
    
    # Identify peaks
    peak_threshold = 0.1 * np.max(abs_fft)
    peak_indices = np.where(abs_fft > peak_threshold)[0]
    peak_freqs = freqs_pos[peak_indices]
    peak_amps = abs_fft[peak_indices]
    
    # Sort peaks by amplitude
    sort_idx = np.argsort(peak_amps)[::-1]
    peak_freqs = peak_freqs[sort_idx]
    peak_amps = peak_amps[sort_idx]
    
    return freqs_pos, abs_fft, peak_freqs, peak_amps