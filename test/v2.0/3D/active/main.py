import datetime
start_time = datetime.datetime.now()
import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Constant
from helmholtz_x.active_flame_x import ActiveFlame
from helmholtz_x.flame_transfer_function_x import n_tau
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer

# Read mesh 
rijke3d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke3d.getAll()
rijke3d.getInfo()

# Define the boundary conditions
import params
boundary_conditions = {1: {'Robin': params.Z_in},  # inlet
                       2: {'Robin': params.Z_out}, # outlet
                       3: {'Neumann'}}             # wall


degree = 1

# Define Speed of sound
c = Constant(mesh, PETSc.ScalarType(1))
c = params.c(mesh)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c , degree = degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

ftf = n_tau(params.n, params.tau)


D = ActiveFlame(mesh, subdomains,
                    params.x_r, params.rho_in, 1., 1., ftf,
                    degree=degree)

D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, np.pi, nev=2, i=0, print_results= False)

omega, p = normalize_eigenvector(mesh, E, 0, degree=1, which='right')

xdmf_writer("Results/p", mesh, p)
    
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
