import numpy as np

def rescale_velocities(system, target_temp, kB=1):
    """
    This function rescales the temperature of an MD simulation

    parameters:
        - system: this will provide a total velocity which will be used to calculate the kinetic energy of the system
        - target_temp: this is the target temperature that the function wll rescale to
        - kB: Boltzman constant, default set to 1

    returns:
        - a rescaled velocity of the whole system to match that of the target temperature
    """

    #define the initial kinetic energy of the system
    KE_initial = system.kinetic_energy
    current_temp = 2/3 * (1/kB) * (1/system.num_atoms) * KE_initial
    scaling_factor = (target_temp / current_temp)**0.5
    system.velocities *= scaling_factor 

    #recalculates KE
    system.kinetic_energy_calculator()
    KE_final = system.kinetic_energy
    current_temp = 2/3 * (1/kB) * (1/system.num_atoms) * KE_final
    return current_temp
