import datetime
start_time = datetime.datetime.now()
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, normalize, XDMFReader
from petsc4py import PETSc
import  params


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 3000

rijke2d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke2d.getAll()
rijke2d.getInfo()

# Define the boundary conditions
boundary_conditions = {4: {'Neumann'},
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Define Speed of sound

c = params.c_DG(mesh, params.x_f, params.c_u, params.c_d)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C


rho = params.rho(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = params.gaussianFunction(mesh, params.x_r, params.a_r)
h = params.gaussianFunction(mesh, params.x_f, params.a_f)

target = 200 * 2 * np.pi # 150 * 2 * np.pi

D = ActiveFlameNT(mesh, subdomains, w, h, rho, 1, 1, params.eta, params.tau, degree=degree)
D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)