import numpy as np
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import pep_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.gmsh_helpers import read_from_msh

# Generate mesh
from rijke_geom import geom_rectangle
if MPI.COMM_WORLD.rank == 0:
    geom_rectangle(fltk=False)

# Read mesh 
mesh, cell_tags, facet_tags = read_from_msh("MeshDir/rijke.msh", cell_data=True, facet_data=True, gdim=2)

# Define the boundary conditions
import params
boundary_conditions = {4: {'Neumann'},
                       3: {'Robin': params.Y_out},
                       2: {'Neumann'},
                       1: {'Robin': params.Y_in}}

# Define Speed of sound
c = dolfinx.Constant(mesh, PETSc.ScalarType(1))

deg = 1

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree =deg)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

eigensolver = pep_solver(A,B,C,np.pi,2,print_results=True)

omega, p = normalize_eigenvector(mesh, eigensolver, 0)

import dolfinx.io
p.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)

