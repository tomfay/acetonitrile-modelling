import numpy as np
from numba import jit
import scipy as sp

@jit(nopython=True)
def calculateNearestImage(d_box,r_i,r_j):
    r_ij = r_i - r_j
    r_ij_NI = r_ij - d_box * np.floor(r_ij/d_box)
    return r_ij_NI

@jit(nopython=True)
def getCentroidVec(positions,w):
    return np.sum((positions.T)*w,axis=1)

# @jit(forceobj=True,nopython=True)
@jit(forceobj=True)
def getCentroidVecs(positions,molecule_inds,w):
    n_mol = len(w)
    centroid_positions = np.zeros((n_mol,3))
    for i in range(0,n_mol):
        centroid_positions[i,:] = getCentroidVec(positions[molecule_inds[i],:],w[i])
    return centroid_positions

# @jit
# def calculateGrFrame(positions,group_A_in,group_B_in,n_mol,d_box,r_min,n_r,dr):
#     g_r_frame = np.zeros(n_r)
#     if group_A_in == []:
#         group_A = list(range(0,n_mol))
#
#     for i in group_A:
#         x_i = positions[i,:]
#         if group_A_in == []:
#             group_B = list(range(0,i))
#         else:
#             group_B = group_B_in.copy()
#             try:
#                 group_B.remove(i)
#             except:
#                 group_B = group_B
#         for j in group_B:
#             x_j = positions[j,:]
#             x_ij = calculateNearestImage(d_box,x_i,x_j)
#             r_ij = np.linalg.norm(x_ij)
#             n = int(np.floor((r_ij-r_min)/dr))
#             if n < n_r :
#                 g_r_frame[n] += 1
#
#     return g_r_frame

@jit(nopython=True)
def calculateGrFrame(positions,n_mol,d_box,r_min,n_r,dr):
    g_r_frame = np.zeros(n_r)


    for i in range(0,n_mol):
        x_i = positions[i,:]

        for j in range(0,i):
            x_j = positions[j,:]
            x_ij = calculateNearestImage(d_box,x_i,x_j)
            r_ij = np.linalg.norm(x_ij)
            n = int(np.floor((r_ij-r_min)/dr))
            if n < n_r :
                g_r_frame[n] += 1

    return g_r_frame

@jit(nopython=True)
def calculateDipoleVec(positions,charges):
    return np.sum((positions.T)*charges , axis=1)

# @jit(forceobj=True,nopython=True)
@jit(forceobj=True)
def getDipoleVecs(positions,molecule_inds,charges):
    n_mol = len(molecule_inds)
    dipole_vecs = np.zeros((n_mol,3))
    for i in range(0,n_mol):
        dipole_vecs[i,:] = calculateDipoleVec(positions[molecule_inds[i],:],charges[molecule_inds[i]].T)
        # print(dipole_vecs[i,:])
    return dipole_vecs

# @jit(forceobj=True,nopython=True)
@jit(forceobj=True)
def getCentroidForces(forces,molecule_inds):
    n_mol = len(molecule_inds)
    centroid_forces = np.zeros((n_mol,3))
    for i in range(0,n_mol):
        centroid_forces[i,:] = np.sum(forces[molecule_inds[i],:],axis=0)
        # print(dipole_vecs[i,:])
    return centroid_forces

@jit(nopython=True)
def calculateGKrFrame(positions,n_mol,d_box,r_min,n_r,dr,dipole_vecs):
    g_K_r_frame = np.zeros(n_r)
    g_r_frame = np.zeros(n_r)
    N = positions.shape[0]
    for i in range(0,N):
        x_i = positions[i,:]

        for j in range(0,i):
            x_j = positions[j,:]
            x_ij = calculateNearestImage(d_box,x_i,x_j)
            r_ij = np.linalg.norm(x_ij)
            n = int(np.floor((r_ij-r_min)/dr))
            if n < n_r :
                g_r_frame[n] += 1.0
                g_K_r_frame[n] += dipole_vecs[i,:].dot(dipole_vecs[j,:])

    return g_r_frame, g_K_r_frame

@jit(nopython=True)
def calculateGKrForceEstimatorFrame(positions,n_mol,d_box,r_min,n_r,dr,dipole_vecs,forces):
    g_K_r_frame = np.zeros((3,n_r))
    g_r_frame = np.zeros((3,n_r))
    b = 0.0
    N = positions.shape[0]
    for i in range(0,N):
        x_i = positions[i,:]

        for j in range(0,i):
            x_j = positions[j,:]
            x_ij = calculateNearestImage(d_box,x_i,x_j)
            r_ij = np.linalg.norm(x_ij)
            n = int(np.floor((r_ij-r_min)/dr))
            mu_i_mu_j = dipole_vecs[i,:].dot(dipole_vecs[j,:])
            if n < n_r :
                g_r_frame[0,n] += 1.0
                g_K_r_frame[0,n] += mu_i_mu_j

            b = 0.5*(forces[i,:]-forces[j,:]).dot(x_ij) / (r_ij**3)
            n = int(np.ceil((r_ij-r_min-0.5*dr)/dr))+1
            # H(r_ij-r)
            g_r_frame[1,0:n] += b
            # H(r-r_ij)
            g_r_frame[2,n::] += -b
            g_K_r_frame[1,0:n] += b*mu_i_mu_j
            g_K_r_frame[2,n::] += -b*mu_i_mu_j



    return g_r_frame, g_K_r_frame

class DipoleCalculator:

    def __init__(self):
        self.molecule_inds = ()
        self.charges = np.empty((0,1))
        self.n_atoms = 0
        self.positions = []
        self.dipole_moment = np.zeros(3)
        self.r_max = 0.0
        self.r_min = 0.0
        self.n_r = 0
        self.g_r = np.zeros(0)
        self.error_g_r = np.zeros(0)
        self.g_K_r = np.zeros(0)
        self.error_g_K_r = np.zeros(0)
        self.r_values = np.zeros(0)
        self.r_upper = np.zeros(0)
        self.r_lower = np.zeros(0)
        self.n_mol = 0
        self.n_atoms = 0
        self.masses = np.zeros(0)
        self.dr = 0.0
        self.w = []
        self.d_box = np.zeros(3)
        self.n_frames = 0
        self.group_A = []
        self.group_B = []
        self.mu_data = []
        self.T = 0.0
        return

    def getCharges(self,system):
        # get the forces in the system
        forces = { force.__class__.__name__ : force for force in system.getForces() }
        # get the nonbonded forces
        nbforce = forces['NonbondedForce']
        # get charges
        self.n_atoms = nbforce.getNumParticles()
        self.charges = np.zeros((self.n_atoms,1))
        for index in range(0,nbforce.getNumParticles()):
            [charge, sigma, epsilon] = nbforce.getParticleParameters(index)
            # print(charge.__dict__)
            self.charges[index] = charge._value

        self.positions = np.zeros((self.n_atoms,3))
        return

    def calculateDipoleMoment(self,simulation):
        self.positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        self.dipole_moment = np.sum(self.positions * self.charges,axis=0)
        return self.dipole_moment

    def getMasses(self,system):
        self.n_atoms = system.getNumParticles()
        self.masses = np.zeros(self.n_atoms)
        for i in range(0,self.n_atoms):
            self.masses[i] = system.getParticleMass(i)._value
        return

    def setupRGrid(self,n_r,r_min,r_max):
        self.r_min = r_min
        self.r_max = r_max
        self.n_r = n_r
        self.dr = (self.r_max - self.r_min)/float(self.n_r)
        self.r_lower = self.r_min + self.dr*np.arange(0,self.n_r)
        self.r_upper = self.r_lower + self.dr
        self.r_values = self.r_lower + 0.5*self.dr
        self.g_r = np.zeros((3,self.n_r))
        self.error_g_r = np.zeros((3,self.n_r))
        self.g_K_r = np.zeros((3,self.n_r))
        self.error_g_K_r = np.zeros((3,self.n_r))
        return

    def setupCentroidWeights(self,weight_method):
        self.w = []
        for i in range(0,self.n_mol):
            if weight_method == "centre of mass":
                w_mol = self.masses[self.molecule_inds[i]]
                w_mol = w_mol * (1.0/np.sum(w_mol))
                self.w.append(w_mol)
            elif weight_method == "centroid":
                n_i = len(self.molecule_inds[i])
                w_mol = np.ones(n_i) * (1.0 / n_i)
                self.w.append(w_mol)
        return

    def getBoxDimensions(self,context):
        v_box = (context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True))._value
        self.d_box = np.diag(v_box)
        return

    def setupDipoleCorrelationEstimator(self,simulation,n_r,r_min,r_max,weight_method):
        # get charges, masses and molecule init_indices
        self.getCharges(simulation.system)
        self.getMasses(simulation.system)
        self.molecule_inds = list(map(list,(simulation.context.getMolecules())))
        self.n_mol = len(self.molecule_inds)
        self.T = simulation.integrator.getTemperature()._value

        # set up grid for calculation of g(r) and g_K(r)
        self.setupRGrid(n_r,r_min,r_max)
        self.setupCentroidWeights(weight_method)
        self.getBoxDimensions(simulation.context)

        return





    def calculateGrCentroid(self,positions):
        g_r_frame = np.zeros(self.n_r)
        if self.group_A == []:
            group_A = list(range(0,self.n_mol))

        for i in group_A:
            x_i = self.getCentroidPosition(positions,i)
            if self.group_A == []:
                group_B = list(range(0,i))
            else:
                group_B = self.group_B.copy()
                try:
                    group_B.remove(i)
                except:
                    group_B = group_B
            for j in group_B:
                x_j = self.getCentroidPosition(positions,j)
                x_ij = calculateNearestImage(self.d_box,x_i,x_j)
                r_ij = np.linalg.norm(x_ij)
                n = int(np.floor((r_ij-self.r_min)/self.dr))
                if n < self.n_r :
                    g_r_frame[n] += 1

        return g_r_frame

    def getCentroidPosition(self,positions,i):
        return np.sum(positions[self.molecule_inds[i],:].T*self.w[i],axis=1)

    def getCentroidPositions(self,positions):
        centroid_positions = np.zeros((self.n_mol,3))
        for i in range(0,self.n_mol):
            centroid_positions[i,:] = self.getCentroidPosition(positions,i)
        return centroid_positions

    def addFrame(self,simulation):
        positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value

        # centroid_positions = self.getCentroidPositions(positions)
        centroid_positions = getCentroidVecs(positions,self.molecule_inds,self.w)
        dipole_vecs = getDipoleVecs(positions,self.molecule_inds,self.charges)
        centroid_forces = getCentroidForces(forces,self.molecule_inds)
        # self.g_r[0,:] = self.g_r[0,:] + self.calculateGrCentroid(positions)
        # self.g_r[0,:] = self.g_r[0,:] + calculateGrFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr)

        # g_r_frame, g_K_r_frame = calculateGKrFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr,dipole_vecs)
        g_r_frame, g_K_r_frame = calculateGKrForceEstimatorFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr,dipole_vecs,centroid_forces)
        self.mu_data.append(list(np.sum(dipole_vecs,axis=0)))
        self.g_r[:,:] = self.g_r[:,:] + g_r_frame
        self.g_K_r[:,:] = self.g_K_r[:,:] + g_K_r_frame
        self.n_frames = self.n_frames + 1
        return

    def normaliseGr(self,simulation):
        T = simulation.integrator.getTemperature()._value

        k_B = 8.31446261815324e-3 # in kJ/mol.K
        beta = 1.0/(k_B * T)
        V = np.prod(self.d_box)
        N = self.n_mol
        self.g_r[0,:] = self.g_r[0,:] * (4.0*V/(N*(N)*self.n_frames*np.pi*self.dr)) * (1.0/(self.r_values*self.r_values))
        self.g_r[1,:] = 1.0 - self.g_r[1,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        self.g_r[2,:] = -self.g_r[2,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        self.g_K_r[0,:] = self.g_K_r[0,:] * (4.0*V/(N*(N)*self.n_frames*np.pi*self.dr)) * (1.0/(self.r_values*self.r_values))
        self.g_K_r[1,:] =  -self.g_K_r[1,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        self.g_K_r[2,:] =  -self.g_K_r[2,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        return

    def writeGrData(self,path):
        T = self.T
        k_B = 8.31446261815324e-3 # in kJ/mol.K
        beta = 1.0/(k_B * T)
        V = np.prod(self.d_box)
        N = self.n_mol
        g_r_file = open(path,'w')
        g_r_file.write("T="+str(T)+"K \n")
        g_r_file.write("V="+str(V)+"nm^3 \n")
        g_r_file.write("N="+str(N)+"\n")
        g_r_file.write("beta="+str(beta)+" (kJ/mol)^-1 \n")
        g_r_file.write("n_frames="+str(self.n_frames)+" \n")
        g_r_file.write("r(nm),g(r)[binning],g(r)[force inf],g(r)[force 0],g_K(r)[binning],g_K(r)[force inf],g_K(r)[force 0]")
        g_r = np.zeros((3,self.n_r))
        g_K_r = np.zeros((3,self.n_r))
        g_r[0,:] = self.g_r[0,:] * (4.0*V/(N*(N)*self.n_frames*np.pi*self.dr)) * (1.0/(self.r_values*self.r_values))
        g_r[1,:] = 1.0 - self.g_r[1,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        g_r[2,:] = -self.g_r[2,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        g_K_r[0,:] = self.g_K_r[0,:] * (4.0*V/(N*(N)*self.n_frames*np.pi*self.dr)) * (1.0/(self.r_values*self.r_values))
        g_K_r[1,:] =  -self.g_K_r[1,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        g_K_r[2,:] =  -self.g_K_r[2,:] * (4.0*beta*V/(N*(N)*self.n_frames*np.pi))
        for i in range(0,self.n_r):
            g_r_data = list(g_r[:,i])
            g_K_r_data = list(g_K_r[:,i])
            data_str = str(self.r_values[i])+","+",".join(map(str,g_r_data)) + "," + ",".join(map(str,g_K_r_data))
            g_r_file.write(data_str+"\n")

        g_r_file.flush()
        g_r_file.close()
        return

    def writeMuData(self,path):
        n_data = len(self.mu_data)
        file = open(path,'w')
        file.write("frame,mu_x(qe nm),mu_y(qe nm),mu_z(qe nm)\n")
        for i,mu_frame in enumerate(self.mu_data):
            data_str = str(i)+","+",".join(map(str,mu_frame))+"\n"
            file.write(data_str)

        file.close()

        return

    def addFrameDipoleOnly(self,simulation):
        # positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        # forces = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value

        # centroid_positions = self.getCentroidPositions(positions)
        # centroid_positions = getCentroidVecs(positions,self.molecule_inds,self.w)
        # dipole_vecs = getDipoleVecs(positions,self.molecule_inds,self.charges)
        # centroid_forces = getCentroidForces(forces,self.molecule_inds)
        # self.g_r[0,:] = self.g_r[0,:] + self.calculateGrCentroid(positions)
        # self.g_r[0,:] = self.g_r[0,:] + calculateGrFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr)

        # g_r_frame, g_K_r_frame = calculateGKrFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr,dipole_vecs)
        # g_r_frame, g_K_r_frame = calculateGKrForceEstimatorFrame(centroid_positions,self.n_mol,self.d_box,self.r_min,self.n_r,self.dr,dipole_vecs,centroid_forces)
        # self.mu_data.append(list(np.sum(dipole_vecs,axis=0)))
        self.mu_data.append(list(self.calculateDipoleMoment(simulation)))
        self.mu_data.append(list(calculateDipoleVec(simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value,self.charges.T)))
        # self.g_r[:,:] = self.g_r[:,:] + g_r_frame
        # self.g_K_r[:,:] = self.g_K_r[:,:] + g_K_r_frame
        self.n_frames = self.n_frames + 1
        return
