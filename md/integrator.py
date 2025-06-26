import numpy as np

def velocity_verlet(system, force_func, dt):
    mass = system.mass
    r = system.positions
    v = system.velocities
    f = system.forces
