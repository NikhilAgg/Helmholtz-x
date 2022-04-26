from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import eps_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.geometry_pkgx.xdmf_utils import XDMFReader
from dolfinx.io import XDMFFile
from dolfinx.fem import Constant
import numpy as np
import datetime
start_time = datetime.datetime.now()

import params

# Read mesh 
micca = XDMFReader("MeshDir/micca")
mesh, subdomains, facet_tags = micca.getAll()

print(micca.getNumberofCells())
# ________________________________________________________________________________
boundary_conditions = {1: 'Neumann',
                       2: 'Neumann',
                       3: 'Neumann',
                       4: 'Neumann',
                       5: 'Neumann',
                       6: 'Neumann',
                       7: 'Neumann',
                       8: 'Neumann',
                       9: 'Neumann',
                       10: 'Neumann',
                       11: 'Neumann',
                       12: 'Master',
                       13: 'Slave'}

degree = 1

target_dir = PETSc.ScalarType(3000)
c = params.c(mesh)

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions,c, degree=degree)
matrices.assemble_A()
matrices.assemble_C()

print(matrices.A.getSizes())

A = matrices.A
C = matrices.C

# set the bloch elements
b = 1
B = 16

from helmholtz_x.helmholtz_pkgx.bloch_operator import Blochifier

bloch_matrices = Blochifier(geometry=micca, boundary_conditions=boundary_conditions, BlochNumber=B, passive_matrices=matrices)
bloch_matrices.blochify_A()
bloch_matrices.blochify_C()

A_petsc_bloch = bloch_matrices.A
C_petsc_bloch = bloch_matrices.C
print("A SIZE: ", A_petsc_bloch.getSizes())
BN_petsc = bloch_matrices.remapper
eigensolver = eps_solver(A_petsc_bloch,C_petsc_bloch,target_dir**2,nev = 5, print_results=True)

omega_1, p_1 = normalize_eigenvector(mesh, eigensolver, 0, mpc=BN_petsc)
omega_2, p_2 = normalize_eigenvector(mesh, eigensolver, 1, mpc=BN_petsc)
omega_3, p_3 = normalize_eigenvector(mesh, eigensolver, 2, mpc=BN_petsc)


print(f"Eigenvalues 1 -> {omega_1:.3f} | Eigenfrequencies ->  {omega_1/(2*np.pi):.3f}")
print(f"Eigenvalues 2 -> {omega_2:.3f} | Eigenfrequencies ->  {omega_2/(2*np.pi):.3f}")
print(f"Eigenvalues 3 -> {omega_3:.3f} | Eigenfrequencies ->  {omega_3/(2*np.pi):.3f}")

# Save eigenvectors

p_1.name = "P_1_Direct"
p_2.name = "P_2_Direct"
p_3.name = "P_3_Direct"


with XDMFFile(MPI.COMM_WORLD, "Results/p_1.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p_1)
with XDMFFile(MPI.COMM_WORLD, "Results/p_2.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p_2)
with XDMFFile(MPI.COMM_WORLD, "Results/p_3.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p_3)
# ________________________________________________________________________________

print("Total Execution Time: ", datetime.datetime.now()-start_time)
