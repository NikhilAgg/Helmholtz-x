import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.active_flame_x import ActiveFlame
from helmholtz_x.flame_transfer_function_x import n_tau
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer
from gmshio import model_to_mesh
import gmsh

# Generate mesh
from rijke_geom import geom_model
if MPI.COMM_WORLD.rank == 0:
    model = geom_model(fltk=False)

# Read mesh 

model_rank = 0
mesh, subdomains, facet_tags = model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank)

# Define the boundary conditions
import params
boundary_conditions = {4: {'Neumann'},
                       3: {'Robin': params.Z_out},
                       2: {'Neumann'},
                       1: {'Robin': params.Z_in}}

degree = 1

# Define Speed of sound
# c = dolfinx.Constant(mesh, PETSc.ScalarType(1))
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
