import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from md.system import MolecularSystem
from md.integrator import velocity_verlet
from md.forces import lennard_jones
from md.thermostat import *


num_atoms = 100
box_size = 10
num_steps = 100
dt = 0.005
temperature = 1
output_xyz = 'lj_tractory.xyz'
output_energy = 'lj_energy.txt'

#initilize the system
system = MolecularSystem(num_atoms=num_atoms, box_size=box_size)
system.randomize_positions(min_dist=0.8)
system.randomize_velocities(temp=temperature)
system.kinetic_energy_calculator()

#compute initial forces
system.forces, _ = lennard_jones(system.positions, system.box_size)

#prepare output files
os.makedirs("outputs", exist_ok=True)
output_xyz = open(f'outputs/{output_xyz}', 'w')
energy_file = open(f'outputs/{output_energy}', 'w')

def write_xyz_step(xyz_file, positions, comment=""):
    xyz_file.write(f'{len(positions)}\n{comment}\n')
    for pos in positions:
        xyz_file.write(f'Atom {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n')

#running the simulation
for step in range(num_steps):
    potential = velocity_verlet(system, lennard_jones, dt)

    kinetic = 0.5 * np.sum(system.mass[:, np.newaxis] * system.velocities**2)
    total_energy = kinetic + potential

    #save every 10 steps
    if step % 10 == 0:
        write_xyz_step(output_xyz, system.positions, f"Step {step}")
        energy_file.write(f"{step}\t{kinetic:.3f}\t{potential:.3f}\t{total_energy:.3f}\n")
        #rescale temperature to be what we want (using berendsen_thermostat)
        _ = berendsen_thermostat(system, 2, 1) #will rescale to have a temperature of 10

    if step % 100 == 0:
        #rescale temperature to be what we want (using rescale_velocities)
        #_ = rescale_velocities(system, 5) #will rescale to have a temperature of 10
        print(f"Step {step}: E_KE = {kinetic:.3f}, E_V = {potential:.3f}, E_total = {total_energy:3f}\n")

output_xyz.close()
energy_file.close()
print("simulation complete Files saved in 'outputs/'")

data = np.loadtxt('outputs/lj_energy.txt')
plt.plot(data[:,0], data[:,1], color='blue', label='Kinetic')
plt.plot(data[:,0], data[:,2], color='red', label='Potential')
plt.plot(data[:,0], data[:,3], color='purple', label='Total')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.title('Energy vs Time')
plt.tight_layout()
plt.show()




