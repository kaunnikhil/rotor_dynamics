import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import qmc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\cv_projects_2\rotor_dynamics\src")))
from fem.elements import ShaftElement, DiskElement
from fem.assembly import RotorAssembly
from fem.solver import RotorSolver

def evaluate_rotor(d_i, d_o, L_total, m_disk, k_bearing):
    
 #  returns the first two natural frequencies for a given set of geometrical an physical parameters

    # parameters
    E = 210e9       # youngs modulus for Steel (Pa)
    G = 81e9        # shear modulus for Steel (Pa)
    rho = 7800      # Density of Steel (kg/m^3)
    num_elements = 10
    L_element = L_total / num_elements
    
    # shaft Elements
    elements = []
    for _ in range(num_elements):
        el = ShaftElement(L=L_element, d_o=d_o, d_i=d_i, E=E, G=G, rho=rho)
        elements.append(el)
        
    # create disks (placing 4 disks at nodes 2, 4, 6, 8)
    r_disk = 0.15 # 15cm radius
    Id = 0.25 * m_disk * r_disk**2
    Ip = 0.5 * m_disk * r_disk**2
    
    disks = {
        2: DiskElement(m=m_disk, Id=Id, Ip=Ip),
        4: DiskElement(m=m_disk, Id=Id, Ip=Ip),
        6: DiskElement(m=m_disk, Id=Id, Ip=Ip),
        8: DiskElement(m=m_disk, Id=Id, Ip=Ip)
    }
    
    # bearings (at ends: node 0 and node 10)
    bearings = {
        0: {'kyy': k_bearing, 'kzz': k_bearing},
        10: {'kyy': k_bearing, 'kzz': k_bearing}
    }
    
    # assemble and Solve
    assembly = RotorAssembly(elements, disks, bearings)
    solver = RotorSolver(assembly)
    
    # keeping the rpm = 3000
    freqs = solver.solve_frequencies(omega_rpm=3000, num_modes=4)
    
    # return the first two fundamental frequencies
    return freqs[0], freqs[1]

def generate_dataset(num_samples=2000):
    
    #using Latin Hypercube Sampling to generate dataset    
    #parameter Bounds [Lower Bound, Upper Bound]
    bounds = {
        'd_i': [0.010, 0.025],       # inner diameter: 10mm to 25mm
        'd_o': [0.030, 0.050],       # outer diameter: 30mm to 50mm
        'L_total': [0.8, 1.5],       # Total length: 0.8m to 1.5m
        'm_disk': [2.0, 10.0],       # Disk mass: 2kg to 10kg
        'k_bearing': [1e5, 1e7]      # Bearing stiffness: 10^5 to 10^7 N/m
    }
    
    # LHS Sampler (5 dimensions for our 5 parameteers)
    num_params = len(bounds)
    sampler = qmc.LatinHypercube(d=num_params, seed=24)#d:dimension 
    sample_points = sampler.random(n=num_samples)
    
    # scaling lhs samples to parameter bounds
    lower_bounds = np.array([b[0] for b in bounds.values()])
    upper_bounds = np.array([b[1] for b in bounds.values()])
    scaled_samples = qmc.scale(sample_points, lower_bounds, upper_bounds)#2000 cross 5
    
    # Initialize lists to hold results
    results = []
    
    # Run the FEM solver for every sample
    for i in range(num_samples):
        d_i, d_o, L_total, m_disk, k_bearing = scaled_samples[i]
        
        # Ensure d_i is strictly less than d_o to avoid negative thickness
        if d_i >= d_o * 0.9:
            d_i = d_o * 0.8 
            
        try:
            f1, f2 = evaluate_rotor(d_i, d_o, L_total, m_disk, k_bearing)
            
            results.append({
                'd_i': d_i,
                'd_o': d_o,
                'L_total': L_total,
                'm_disk': m_disk,
                'k_bearing': k_bearing,
                'freq_1_hz': f1,
                'freq_2_hz': f2
            })
        except Exception as e:
            # physically invalid geometry skipped
            continue
            
        if (i+1) % 200 == 0:
            print(f"Completed {i+1} / {num_samples} simulations.")
            
    df = pd.DataFrame(results)

    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\cv_projects_2\rotor_dynamics\data\raw")), exist_ok=True)
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r"D:\cv_projects_2\rotor_dynamics\data\raw\rotor_dataset.csv"))
    
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    generate_dataset(num_samples=2000)