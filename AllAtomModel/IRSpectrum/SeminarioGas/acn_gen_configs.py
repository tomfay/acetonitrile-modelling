import numpy as np
# from simtk.unit import *
# from simtk.openmm.app import *
# from simtk.openmm import *
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import timeit
import dipole

start_time = timeit.default_timer()
max_time = 36.0*60.0*60.0 - 10.0*60.0

mu_data_file_pathbase = "mu"
# read pdb file of positions
pdb = PDBFile("acn.pdb")
n_frames = pdb.getNumFrames()
T = 298.0
dt = 0.002
n_eq = 1000
n_step = 200000
d_box = 10.0
pdb.topology.setUnitCellDimensions((d_box,d_box,d_box))

# read forcefield xml file
forcefield = ForceField("acn_nocorr_gasq.xml")

# create the system from the topology and forcefield files
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)




# add barostat
# system.addForce(MonteCarloAnisotropicBarostat((1, 1, 1)*bar, 300*kelvin, True,True,True,25))
# system.addForce(MonteCarloBarostat(1.0*bar, T*kelvin,20))




# create integrator object
integrator = LangevinMiddleIntegrator(T*kelvin, 2.0/picosecond, dt*picoseconds)
# integrator = VerletIntegrator(dt*picoseconds)

# set up the platform
# platform = Platform.getPlatformByName('CUDA') #change it to CPU if no GPUs are available

# set up the simulation & platform
platform = Platform.getPlatformByName('CPU')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.getPositions(frame=0))
simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
        potentialEnergy=True, temperature=True,density=True,volume=True))

simulation.step(n_eq)
# set up reporters
simulation.reporters.append(PDBReporter('prod_traj0.pdb', 1000))
simulation.step(n_step)



print("Time elapsed:",timeit.default_timer()-start_time," seconds")
