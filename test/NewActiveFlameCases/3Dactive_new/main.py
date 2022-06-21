import datetime
start_time = datetime.datetime.now()

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Constant
from helmholtz_x.helmholtz_pkgx.active_flame_x_new import ActiveFlame
from helmholtz_x.helmholtz_pkgx.flame_transfer_function_x import n_tau
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.geometry_pkgx.xdmf_utils import XDMFReader

# Read mesh 
RijkeTube3D = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = RijkeTube3D.getAll()
RijkeTube3D.getNumberofCells()

# Define the boundary conditions
import params
boundary_conditions = {1: {'Robin': params.Y_in},  # inlet
                       2: {'Robin': params.Y_out}, # outlet
                       3: {'Neumann'}}             # wall


degree = 2

x_f = params.x_f
x_r = params.x_r

# Define Speed of sound
c = params.c(mesh, x_f[0][2])

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c , degree = degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

ftf = n_tau(params.n, params.tau)



# print(x_f,x_r)

rho = params.rho(mesh, x_f[0][2])
w = params.w(mesh, x_r[0][2])


D = ActiveFlame(mesh, subdomains,
                    w, rho, params.Q_tot, params.U_bulk, ftf, x_r, 
                    degree=degree)

D.assemble_submatrices()
if MPI.COMM_WORLD.rank == 0:
    print("Matrix D assembled")
E = fixed_point_iteration_pep(matrices, D, 1160, nev=2, i=0, print_results= False)

omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')
if MPI.COMM_WORLD.rank == 0:
    print(f"Eigenvalue-> {omega:.3f} | Eigenfrequency ->  {omega/(2*np.pi):.3f}")

import dolfinx.io
p.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Results/p.xdmf", "w") as xdmf:
    p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)