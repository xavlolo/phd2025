import numpy as np

def commutator(H, rho):
    """Calculate the quantum commutator [H, rho] = -i(H·rho - rho·H)"""
    return -1j * (H @ rho - rho @ H)

def rk4_step(H, rho, dt):
    """Fourth-order Runge-Kutta integration step for density matrix evolution"""
    k1 = dt * commutator(H, rho)
    k2 = dt * commutator(H, rho + 0.5 * k1)
    k3 = dt * commutator(H, rho + 0.5 * k2)
    k4 = dt * commutator(H, rho + k3)
    return rho + (k1 + 2*k2 + 2*k3 + k4) / 6