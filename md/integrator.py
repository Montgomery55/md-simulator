import numpy as np

def velocity_verlet(system, force_func, dt):
    """
    performs one timestep using the velocity verlet algorithm

    parameters:
        - system: MolecularSystem
        - force_func, force_functioin(positions, box_size) -> forces and potential_energy
        - dt: timestep size
    """
    #half step velocity update
    system.velocities += dt/2 * system.forces/system.mass

    #full step position update
    system.positions += dt * system.velocities 
    system.apply_periodic_boundary_conditions()

    # calculate forces on new positions
    f_new, potential = force_func(system.positions, system.box_size)
    system.forces = f_new

    #second step velocity update
    system.velocities += dt/2 * system.forces/system.mass

    return potential
