import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from md.system import MolecularSystem
from md.integrator import velocity_verlet
from md.forces import lennard_jones


num_atoms = 100
box_size = 10
num_steps = 500
dt = 0.005
temperature = 1
output_xyz = 'lj_tractory.xyz'
output_energy = 'lj_energy.xyz'

#initilize the system
system = MolecularSystem(num_atoms=num_atoms, box_size=box_size)
system.randomize_positions(min_dist=0.8)
system.randomize_velocities(temp=temperature)

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

    kinetic = 0.5 * system.mass * np.sum(system.velocities**2)
    total_energy = kinetic + potential

    #save every 10 steps
    if step % 10 == 0:
        write_xyz_step(output_xyz, system.positions, f"Step {step}")
        energy_file.write(f"{step}\t{kinetic:.3f}\t{potential:.3f}\t{total_energy:.3f}\n")

    if step % 100 == 0:
        print(f"Step {step}: E_kine = {kinetic:.3f}, E_pot = {potential:.3f}, E_total = {total_energy:3f}\n")

output_xyz.close()
energy_file.close()
print("simulation complete Files saved in 'outputs/'")






