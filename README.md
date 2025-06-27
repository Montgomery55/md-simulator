# `md-simulator`

`md-simulator` runs simple molecular dynamics simulations of charged/neutral ''molecules''
(simple spherical models of molecules).

## USAGE

The MD simulation is initiated by the `MolecularSystem` class.
`MolecularSystem` consist of `.num_atoms`, `.box_size`, `.mass`, and `.charge`.
All of these are initilized with the initilization of the class.

Additional properties, `.positions` (produced by the  `.randomize_positions()` function),
`.velocities` (produced by the `.randomize_velocities()` function), 
`.forces` (produced by the several force functions discussed below),
and `.kinetic_energy` (produced by the `kinetic_energy_calculator()`).

The forces in the system are calculated by two functions, located in `md/forces.py`.
The `lennard_jones()` function produces nonbonded nonelectrostatic interactions between atoms
while the `electrostatic_forces()` function calculates simple electrostatic interactions between the atoms.
Finally, the `noncovalent_potential_and_forces()` calculates the sum of both electrostatic and
the other nonbonded interactions.

`md/integrator.py` provides the function `velocity_verlet()` which performs Verlet integration
for each time step i.e. how the MD simulation progresses in time.

Finally, the `md/thermostat.py` file contains two thermometer functions, `rescale_velocities()`
and `berendsen_thermostat()`.
This provides a way to control the kinetic energy of the system.

An example on how to run a simple MD simulation using this code is given in `examples/lj_has.py`.
