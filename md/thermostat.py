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
        - current_temp: to verify that the temperature rescaling is working
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

def berendsen_thermostat(system, target_temp, dt, tau=1.0, kB=1.0):
    """
    This function rescales the temperature of an MD simulation smoothly, unlike the rescale_velocities function which does it sharply
    This is good for equilibration but bad for production runs

    parameters:
        - system: this will provide a total velocity which will be used to calculate the kinetic energy of the system
        - target_temp: this is the target temperature that the function wll rescale to
        - tau: describes how smoothly the temperature will change (big tau causes slow convergence small tau gives quick convergnce), should range from 0.1 to 1
        - dt: the step at which you want to change the temperature at a given time, should be between 0 and 1 so as to not get a negative sqrt
        - kB: Boltzman constant, default set to 1

    returns:
        - lambda: the rescaling factor that modifies the velocities
        - a rescaled velocity of the whole system to match that of the target temperature
        - current_temp: to verify that the temperature rescaling is working
    """
    current_temp = 2/3 * (1/kB) * (1/system.num_atoms) * system.kinetic_energy
    lambda_factor = (1 + dt / tau * (target_temp / current_temp - 1))**0.25
    system.velocities *= lambda_factor

    system.kinetic_energy_calculator()
    current_temp = 2/3 * (1/kB) * (1/system.num_atoms) * system.kinetic_energy
    return current_temp




