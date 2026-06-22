# this script assembles local element matrices into global M, K, and G matrices.
# and applies boundary conditions (flexible bearings) to the global stiffness matrix.
import numpy as np

class RotorAssembly:
    def __init__(self, shaft_elements, disks, bearings):
        """
        shaft_elements: List of ShaftElement objects.
        disks: Dictionary of {node_index: DiskElement}
        bearings: Dictionary of {node_index: {'kyy': val, 'kzz': val}}
        (kyy and kzz are the bearing stiffness coefficients particularly against the translation only)
        """
        self.shaft_elements = shaft_elements
        self.disks = disks
        self.bearings = bearings
        
        # Calculate system size
        self.num_elements = len(shaft_elements)
        self.num_nodes = self.num_elements + 1
        self.dof_per_node = 4 #u, v, theta1, theta2
        self.total_dof = self.num_nodes * self.dof_per_node
        
        # Initialize global matrices with zeros
        self.M = np.zeros((self.total_dof, self.total_dof))
        self.K = np.zeros((self.total_dof, self.total_dof))
        self.G = np.zeros((self.total_dof, self.total_dof))

    def assemble(self):
        # assemble Shaft Elements (Overlapping 8x8 matrices)
        for i, element in enumerate(self.shaft_elements):
            idx = i * self.dof_per_node
            
            # both the 8x8 local matrices
            m_e = element.get_mass_matrix()
            k_e = element.get_stiffness_matrix()
            
            # add to global matrices (overlapping by 4 DOFs at the nodes)
            self.M[idx:idx+8, idx:idx+8] += m_e
            self.K[idx:idx+8, idx:idx+8] += k_e

        # assemble Disks 
        for node, disk in self.disks.items():
            idx = node * self.dof_per_node
            
            m_d = disk.get_mass_matrix()
            g_d = disk.get_gyroscopic_matrix()
            
            # Disks add mass and gyroscopic effects, but no stiffness
            self.M[idx:idx+4, idx:idx+4] += m_d
            self.G[idx:idx+4, idx:idx+4] += g_d

        # apply boundary conditions of bearings
        for node, bearing_props in self.bearings.items():
            idx = node * self.dof_per_node
            
            kyy = bearing_props.get('kyy', 0)
            kzz = bearing_props.get('kzz', 0)
            
            # add directly to the translational diagonals of the global stiffness matrix
            # Translation Y is DOF 0 relative to the node
            # Translation Z is DOF 1 relative to the node
            self.K[idx, idx] += kyy
            self.K[idx+1, idx+1] += kzz

        return self.M, self.K, self.G