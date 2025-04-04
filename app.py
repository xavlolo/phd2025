#!/usr/bin/env python3
"""
Quantum Spin Network Web Application

A web interface for running and visualizing quantum spin network simulations.
"""

import os
import sys
import json
import base64
import importlib
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation modules
from basis import generate_selective_basis, initialize_superposition_rho
from hamiltonian import generate_hamiltonian_with_selective_k
from web_helpers import evolve_density_matrix, calculate_state_probabilities, run_time_evolution
from partial_trace import partial_trace, analyze_reduced_density_matrix, analyze_entanglement_over_time

# Create Flask app
app = Flask(__name__)

# Make sure output directory exists
os.makedirs('static/output', exist_ok=True)

# Global variables to store simulation results
simulation_results = {
    'plots': [],
    'console_output': [],
    'data': {}
}


def capture_output(func):
    """Decorator to capture console output and return it"""
    from io import StringIO
    import sys

    def wrapper(*args, **kwargs):
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get the output
        output = mystdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        return result, output
    
    return wrapper


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML
    
    Args:
        fig: A matplotlib figure object or a path to an image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        if isinstance(fig, str) and os.path.isfile(fig):
            # If fig is a file path, read the file
            with open(fig, 'rb') as f:
                img_data = f.read()
                img_str = base64.b64encode(img_data).decode('utf-8')
                return img_str
        elif hasattr(fig, 'savefig'):
            # If fig is a matplotlib figure
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            return img_str
        else:
            # Try to convert other types of objects
            print(f"Warning: Unexpected type for fig_to_base64: {type(fig)}")
            return ""
    except Exception as e:
        print(f"Error in fig_to_base64: {str(e)}")
        return ""


@capture_output
def run_superposition_simulation(n_qubits=6, dt=0.01, t_max=50):
    """Run basic superposition simulation"""
    try:
        # Validate parameters
        if n_qubits < 2 or n_qubits > 10:
            raise ValueError(f"Number of qubits must be between 2 and 10, got {n_qubits}")
        if dt <= 0 or dt > 1:
            raise ValueError(f"Time step must be between 0 and 1, got {dt}")
        if t_max <= 0 or t_max > 100:
            raise ValueError(f"Maximum time must be between 0 and 100, got {t_max}")
        
        print(f"Starting superposition simulation with {n_qubits} qubits, dt={dt}, t_max={t_max}")
        print(f"This simulation will run online through the Flask web interface")
        
        # Generate basis states
        basis_states, state_to_idx = generate_selective_basis(
            n_qubits, 
            top_excitations=True, 
            bottom_excitations=True
        )
        
        # Define the superposition state - adjust based on number of qubits
        superposition_dict = {}
        if n_qubits >= 6:
            superposition_dict = {
                (1, 1, 0, 0, 0, 0): 1.0 + 0j,
                (1, 0, 0, 1, 0, 0): 0.5 + 0.5j,
                (1, 0, 0, 0, 0, 1): 0.3 - 0.2j
            }
        elif n_qubits >= 4:
            superposition_dict = {
                (1, 1, 0, 0): 1.0 + 0j,
                (1, 0, 0, 1): 0.5 + 0.5j
            }
        else:  # n_qubits >= 2
            superposition_dict = {
                (1, 1): 1.0 + 0j
            }
        
        print(f"Using superposition of {len(superposition_dict)} states")
        
        # Initialize density matrix
        initial_rho = initialize_superposition_rho(
            basis_states, state_to_idx, n_qubits, superposition_dict
        )
        
        # Generate Hamiltonian with appropriate k_pattern based on number of qubits
        k_pattern = {}
        for i in range(0, n_qubits - 1, 2):
            if i + 1 < n_qubits:
                k_pattern[(i, i+1)] = 1.0
        
        print(f"Using k_pattern: {k_pattern}")
        
        H, eigenvalues = generate_hamiltonian_with_selective_k(
            basis_states, state_to_idx, n_qubits, 
            J_max=1.0, k_pattern=k_pattern, E_site=0.0
        )
        
        # Time evolution
        times = np.arange(0, t_max + dt, dt)
        
        # Track probabilities of specific states
        state_probs = {}
        for state, amp in superposition_dict.items():
            state_probs[state] = []
        
        # Evolve the system
        print("Evolving quantum system...")
        rho = initial_rho.copy()
        for t in times:
            # Calculate probabilities for each state in the superposition
            for state in superposition_dict:
                idx = state_to_idx[state]
                prob = np.real(rho[idx, idx])
                state_probs[state].append(prob)
            
            # Evolve to next time step
            if t < t_max:
                rho = evolve_density_matrix(rho, H, dt)
        
        print("Creating visualization...")
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for state, probs in state_probs.items():
            state_str = '|' + ''.join(str(s) for s in state) + '⟩'
            ax.plot(times, probs, label=state_str)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.set_title('Quantum State Evolution')
        ax.legend()
        ax.grid(True)
        
        # Make sure output directory exists
        os.makedirs('static/output', exist_ok=True)
        
        # Save plot
        plot_path = 'static/output/superposition_evolution.png'
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        # Return data for interactive plotting
        plot_data = {
            'times': times.tolist(),
            'state_probs': {str(state): probs for state, probs in state_probs.items()}
        }
        
        return {
            'plot_path': plot_path,
            'plot_data': plot_data,
            'fig': fig
        }
    except Exception as e:
        import traceback
        print(f"Error in superposition simulation: {str(e)}")
        print(traceback.format_exc())
        raise


@capture_output
def run_partial_trace_analysis(use_nonuniform_k=False):
    """Run analysis of partial trace for the quantum spin network"""
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
    
    # Initialize with first qubit excited
    initial_rho = initialize_superposition_rho(
        basis_states, state_to_idx, n_qubits,
        {(1, 0, 0, 0, 0, 0): 1.0}
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
    
    # Generate Hamiltonian
    H, eigenvalues = generate_hamiltonian_with_selective_k(
        basis_states, state_to_idx, n_qubits, 
        J_max=1.0, k_pattern=k_pattern, E_site=0.0
    )
    
    # Analyze initial reduced density matrix
    reduced_rho_initial, state_map = partial_trace(
        initial_rho, basis_states, [0, 2, 4], n_qubits
    )
    
    # Run detailed analysis of reduced density matrix over time
    sample_times = [0, 12.5, 25, 37.5, 50]
    rdm_plots = analyze_reduced_density_matrix(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        initial_rho, H, 
        qubits_to_trace_out=[0, 2, 4],  # Trace out top qubits
        sample_times=sample_times  # Sample at these times
    )
    
    # Analyze entanglement over time
    entanglement_plot = analyze_entanglement_over_time(
        basis_states, state_to_idx, n_qubits, dt, t_max,
        initial_rho, H
    )
    
    return {
        'rdm_plots': rdm_plots,
        'entanglement_plot': entanglement_plot,
        'k_pattern': k_pattern,
        'pattern_name': pattern_name
    }


@app.route('/')
def index():
    """Render the main page"""
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/output', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    return render_template('index.html')


@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run a simulation based on user input"""
    # Set a reasonable timeout for simulations (30 seconds)
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Simulation timed out")
    
    # Register the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    simulation_type = request.form.get('simulation_type', 'superposition')
    print(f"Received simulation request of type: {simulation_type}")
    
    # Clear previous results
    simulation_results['plots'] = []
    simulation_results['console_output'] = []
    simulation_results['data'] = {}
    
    try:
        if simulation_type == 'superposition':
            try:
                n_qubits = int(request.form.get('n_qubits', 6))
                dt = float(request.form.get('dt', 0.01))
                t_max = float(request.form.get('t_max', 50))
            except ValueError as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid parameter value: {str(e)}'
                })
            
            try:
                result, output = run_superposition_simulation(n_qubits, dt, t_max)
                
                # Store results
                simulation_results['plots'].append({
                    'title': 'Quantum State Evolution',
                    'path': result['plot_path'],
                    'img': fig_to_base64(result['fig'])
                })
                simulation_results['console_output'].append(output)
                simulation_results['data']['superposition'] = result['plot_data']
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Error in superposition simulation: {str(e)}\n{traceback_str}")
                return jsonify({
                    'status': 'error',
                    'message': f'Simulation error: {str(e)}'
                })
            
        elif simulation_type == 'partial_trace':
            use_nonuniform_k = request.form.get('use_nonuniform_k', 'false') == 'true'
            
            try:
                result, output = run_partial_trace_analysis(use_nonuniform_k)
                
                # Store results
                for i, plot_path in enumerate(result.get('rdm_plots', [])):
                    try:
                        fig = plt.figure()
                        img = plt.imread(plot_path)
                        plt.imshow(img)
                        plt.axis('off')
                        title = f"Reduced Density Matrix at t={i*12.5:.2f}"
                        simulation_results['plots'].append({
                            'title': title,
                            'path': plot_path,
                            'img': fig_to_base64(fig)
                        })
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error processing plot {plot_path}: {str(e)}")
                
                # Add entanglement plot if available
                if 'entanglement_plot' in result and result['entanglement_plot']:
                    try:
                        fig = plt.figure()
                        img = plt.imread(result['entanglement_plot'])
                        plt.imshow(img)
                        plt.axis('off')
                        simulation_results['plots'].append({
                            'title': 'Entanglement Measures Over Time',
                            'path': result['entanglement_plot'],
                            'img': fig_to_base64(fig)
                        })
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error processing entanglement plot: {str(e)}")
                
                simulation_results['console_output'].append(output)
                simulation_results['data']['k_pattern'] = result.get('k_pattern', {})
                simulation_results['data']['pattern_name'] = result.get('pattern_name', '')
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Error in partial trace analysis: {str(e)}\n{traceback_str}")
                return jsonify({
                    'status': 'error',
                    'message': f'Simulation error: {str(e)}'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unknown simulation type: {simulation_type}'
            })
        
        # Ensure we're returning valid JSON data
        response_data = {
            'status': 'success',
            'plots': [],
            'console_output': simulation_results['console_output'] if simulation_results['console_output'] else ['No output available']
        }
        
        # Process plots to ensure valid data
        for p in simulation_results['plots']:
            if 'title' in p and 'img' in p:
                # Convert numpy arrays or other non-serializable objects to strings if needed
                img_data = p['img']
                if not isinstance(img_data, str):
                    # If it's not already a string (like a base64 encoded image)
                    # try to convert it to a string representation
                    try:
                        img_data = str(img_data)
                    except Exception as e:
                        print(f"Warning: Could not convert image data to string: {e}")
                        continue
                        
                response_data['plots'].append({
                    'title': p['title'],
                    'img': img_data
                })
        
        # Ensure the response is properly serializable
        try:
            # Test JSON serialization before returning
            json.dumps(response_data)
            # Cancel the timeout alarm
            signal.alarm(0)
            return jsonify(response_data)
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            # Return a simplified response if serialization fails
            # Cancel the timeout alarm
            signal.alarm(0)
            return jsonify({
                'status': 'success',
                'message': 'Simulation completed but results contain non-serializable data',
                'console_output': simulation_results['console_output'] if simulation_results['console_output'] else ['No output available']
            })
    
    except TimeoutError as e:
        # Handle simulation timeout
        signal.alarm(0)  # Cancel the alarm
        print(f"Simulation timed out: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Simulation timed out. Try reducing the simulation parameters (e.g., fewer qubits or shorter time).'
        })
    except Exception as e:
        # Handle other exceptions
        signal.alarm(0)  # Cancel the alarm
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Unexpected error: {str(e)}\n{traceback_str}")
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        })


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


# Add an error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f"Unhandled exception: {str(e)}")
    import traceback
    app.logger.error(traceback.format_exc())
    
    # Return a JSON response for API requests
    if request.path.startswith('/run_simulation'):
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500
    
    # Return an HTML response for other requests
    return f"<h1>Server Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>", 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    try:
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=8888)
    except Exception as e:
        print(f"Error starting Flask app: {str(e)}")
        import traceback
        print(traceback.format_exc())
