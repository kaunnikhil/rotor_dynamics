#below script fetches the Me, Md, K and the G  matrices required in the finite rotor element equation of motion subjected to gyroscopic effects
import numpy as np

class DiskElement:
    #represents a rigid disk attached to a specific node on the shaft.
    # a disk adds mass and inertia but does not add stiffness.

    
    def __init__(self, m, Id, Ip):
        
        """
        m: Mass of the disk in kg
        Id: Diametral moment of inertia in SI 
        Ip: Polar moment of inertia in SI
        """
        self.m = m
        self.Id = Id
        self.Ip = Ip

    def get_mass_matrix(self): #gets the {Md}
        #4 cross 4 mass matrix bcoz disc represents 1 node having 4DOF {y, z, theta-y, theta-z}
        return np.array([
            [self.m, 0, 0, 0],
            [0, self.m, 0, 0],
            [0, 0, self.Id, 0],
            [0, 0, 0, self.Id]
        ])

    def get_gyroscopic_matrix(self): #gets the [G]
        #returns the 4 cross 4 gyroscopic matrix for the disk.
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, self.Ip],
            [0, 0, -self.Ip, 0]
        ])


class ShaftElement:
    """
    represents a Timoshenko hollow shaft finite element.
    Calculates the 8 cross 8 matrices for a 2-node, 4 DOF per node element.
    """
    def __init__(self, L, d_o, d_i, E, G, rho):
        self.L = L
        self.d_o = d_o
        self.d_i = d_i #
        self.E = E #youngs modulus 
        self.G = G #shear modulus
        self.rho = rho #density
        
        # cross sectional properties
        self.A = (np.pi / 4) * (self.d_o**2 - self.d_i**2)
        self.I = (np.pi / 64) * (self.d_o**4 - self.d_i**4)#this is the bending inertia
        
        # kappa shear shape factor for a hollow circular cross-section where we use the exact formula, not 0.5 approx'n
        # it represents the part in CSA that effectively resists shear hence shear stiffness = kappa*G*A
        m_ratio = self.d_i / self.d_o
        self.kappa = 6 * (1 + m_ratio**2)**2 / (7 + 34*m_ratio**2 + 7*m_ratio**4)
        
        # phi transverse shear effect parameter
        self.Phi = (12 * self.E * self.I) / (self.kappa * self.A * self.G * self.L**2)
        
        # mass per unit length= density * CSA
        self.mu = self.rho * self.A

    def get_stiffness_matrix(self): #gets the {Ke}
        """Returns the 8x8 symmetric stiffness matrix for a Timoshenko beam."""
        L = self.L
        EI = self.E * self.I
        P = self.Phi
        
        # constant multiplier
        c = EI / (L**3 * (1 + P))
        
        # components based on Nelson Timoshenko formulation
        K = np.zeros((8, 8))
        
        # Y-Z decoupled bending 

        K[0,0] = 12;          K[0,3] = 6*L;                 K[0,4] = -12;         K[0,7] = 6*L
        K[3,0] = K[0,3];      K[3,3] = L**2 * (4 + P);      K[3,4] = -6*L;        K[3,7] = L**2 * (2 - P)
        K[4,0] = K[0,4];      K[4,3] = K[3,4];              K[4,4] = 12;          K[4,7] = -6*L
        K[7,0] = K[0,7];      K[7,3] = K[3,7];              K[7,4] = K[4,7];      K[7,7] = L**2 * (4 + P)
        
 
        K[1,1] = 12;          K[1,2] = -6*L;                K[1,5] = -12;         K[1,6] = -6*L
        K[2,1] = K[1,2];      K[2,2] = L**2 * (4 + P);      K[2,5] = 6*L;         K[2,6] = L**2 * (2 - P)
        K[5,1] = K[1,5];      K[5,2] = K[2,5];              K[5,5] = 12;          K[5,6] = 6*L
        K[6,1] = K[1,6];      K[6,2] = K[2,6];              K[6,5] = K[5,6];      K[6,6] = L**2 * (4 + P)
        
        return c * K

    def get_mass_matrix(self):#gets the {Me}
        #returns the 8 cros 8 translational mass matrix [M_e]. 
        L = self.L
        P = self.Phi
        c = (self.mu * L) / 420
        M = np.zeros((8, 8))
        
        # Y-Z decoupled
        M[0,0] = 156; M[0,3] = 22*L; M[0,4] = 54; M[0,7] = -13*L
        M[3,0] = M[0,3]; M[3,3] = 4*L**2; M[3,4] = 13*L; M[3,7] = -3*L**2
        M[4,0] = M[0,4]; M[4,3] = M[3,4]; M[4,4] = 156; M[4,7] = -22*L
        M[7,0] = M[0,7]; M[7,3] = M[3,7]; M[7,4] = M[4,7]; M[7,7] = 4*L**2
        
        M[1,1] = 156; M[1,2] = -22*L; M[1,5] = 54; M[1,6] = 13*L
        M[2,1] = M[1,2]; M[2,2] = 4*L**2; M[2,5] = -13*L; M[2,6] = -3*L**2
        M[5,1] = M[1,5]; M[5,2] = M[2,5]; M[5,5] = 156; M[5,6] = 22*L
        M[6,1] = M[1,6]; M[6,2] = M[2,6]; M[6,5] = M[5,6]; M[6,6] = 4*L**2
        
        return c * M