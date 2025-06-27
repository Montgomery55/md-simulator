import numpy as np

class MolecularSystem:
    def __init__(self, num_atoms, box_size, mass=1.0):
        self.num_atoms = num_atoms
        self.box_size = box_size
        self.mass = mass #assumes same mass for all items, TODO: make it so different masses are possible

        self.positions = np.zeros((num_atoms, 3))
        self.velocities = np.zeros((num_atoms, 3))
        self.forces = np.zeros((num_atoms, 3))
        self.kinetic_energy = 0

    def randomize_positions(self, min_dist=0.5):
        #randomizes all atom positions so they are all at least 0.5 units apart
        positions = []
        while len(positions) < self.num_atoms:
            potential = np.random.rand(3) * self.box_size
            if all(np.linalg.norm(potential - p) > min_dist for p in positions):
                positions.append(potential)
        self.positions = positions

    def randomize_velocities(self, temp=1.0, kB=1):
        # randomizes velocities based off of a boltzman distribution at a given temperature
        stddev = np.sqrt(kB * temp / self.mass)
        self.velocities = np.random.normal(0, stddev, (self.num_atoms, 3))

        com_velocity = np.mean(self.velocities, axis=0)
        self.velocities -= com_velocity

    def apply_periodic_boundary_conditions(self):
        #wraps positions back into the box if they leave the box, minimum image convention
        self.positions = [pos % self.box_size for pos in self.positions]

    def kinetic_energy_calculator(self):
        v2 = np.sum(self.velocities**2, axis=1)
        kinetic_energy = 0.5 * np.sum(self.mass * v2)
        self.kinetic_energy = kinetic_energy
