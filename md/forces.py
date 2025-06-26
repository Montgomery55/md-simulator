import numpy as np

def lennard_jones(positions, box_size, epsilon=1.0, sigma=1.0, r_cutoff=2.5):
    """
    computes lennard-jones forces using a minimum image convention

    parameters:
        - positions: (N,3) array of atomic positions
        - box_size: simulation box size (cube)
        - epsilon, sigma: lennard-jones parameters
        - r_cutoff: the distance at which the interaction is damped to zero

    Returns:
        - forces (N, 3) array of the forces on each atom
        - total_potential: total lennard-jones potential energy
    """
    N = np.array(positions).shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    positions = positions

    box = np.array([box_size]*3)

    for i in range(N):
        for j in range(i+1, N):
            rij = positions[i] - positions[j]
            rij -= box * np.round(rij / box)
            r2 = np.linalg.norm(rij)
            
            if r2 > r_cutoff:
                potential_energy += 4 * epsilon * ((sigma / r2)**12 - (sigma / r2)**2)
                force_scalar = -4 * epsilon * ((6*sigma**6/r2**7) - (12*sigma**12/r2**13))
                forces[i] += rij * force_scalar
                forces[j] -= rij * force_scalar

    return forces, potential_energy
