import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlame
from helmholtz_x.helmholtz_pkgx.flame_transfer_function_x import n_tau
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.gmsh_helpers import read_from_msh

# Generate mesh
from rijke_geom import geom_rectangle
if MPI.COMM_WORLD.rank == 0:
    geom_rectangle(fltk=False)

# Read mesh 
mesh, subdomains, facet_tags = read_from_msh("MeshDir/rijke.msh", cell_data=True, facet_data=True, gdim=2)

# Define the boundary conditions
import params
boundary_conditions = {4: {'Neumann'},
                       3: {'Robin': params.Y_out},
                       2: {'Neumann'},
                       1: {'Robin': params.Y_in}}

degree = 1

# Define Speed of sound
c = dolfinx.Constant(mesh, PETSc.ScalarType(1))


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
                    degree=degree, comm = MPI.COMM_WORLD)

D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, np.pi, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=1, which='right')
print(omega)
