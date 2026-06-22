# this script solves the rotor dynamics eigenvalue problem using the State Space formulation.
#    Extracts natural frequencies for Campbell diagrams.
import numpy as np
from scipy.linalg import eig

class RotorSolver:
    def __init__(self, assembly):
        
        #assembly: assembled RotorAssembly object.
        self.M, self.K, self.G = assembly.assemble()
        self.total_dof = self.M.shape[0]
        self.M_inv = np.linalg.inv(self.M)

    def solve_frequencies(self, omega_rpm, num_modes=10):
        """
        solves for the natural frequencies at a specific rotational speed.
        omega_rpm= Rotational speed rpm
        num_modes= no. of lowest frequencies to return
        """
        omega_rad = omega_rpm * (np.pi / 30)
        
        # we use the state-space matrix to solve the quadratic eigenvalue problem
        # A_sys = [    0 (44x44)               I (44x44)       ]
        #         [ -M_inv * K (44x44)    -M_inv * (omega*G) ]
        
        I = np.eye(self.total_dof)
        Z = np.zeros((self.total_dof, self.total_dof))
        
        top_row = np.hstack((Z, I))
        bottom_row = np.hstack((
            -self.M_inv @ self.K, 
            -self.M_inv @ (omega_rad * self.G)
        ))
        
        A_sys = np.vstack((top_row, bottom_row))
        
        # Solve the eigenvalue problem
        eigenvalues, _ = eig(A_sys)
        
        # Extract the natural frequencies (positive imaginary parts)
        # We divide by 2*pi to convert from rad/s back to Hz
        frequencies_rad = np.imag(eigenvalues)
        frequencies_hz = frequencies_rad[frequencies_rad > 1e-5] / (2 * np.pi)
        
        # Sort frequencies from lowest to highest
        frequencies_hz = np.sort(frequencies_hz)
        
        # Because eigenvalues come in complex conjugate pairs, 
        # we take every second value to avoid duplicates if G=0 bcoz without the gyroscopic effect the forward and backward whirl becomes identical and there is no frequency splitting,
        # but with G>0, forward and backward whirl split into distinct frequencies.
        # We will return the lowest 'num_modes' physical frequencies.
        return frequencies_hz[:num_modes]