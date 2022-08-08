import numpy as np
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.eigensolvers_x import eps_solver
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.dolfinx_utils import write_xdmf_mesh ,XDMFReader, xdmf_writer

# Generate mesh
from rijke_geom import geom_rectangle
if MPI.COMM_WORLD.rank == 0:
    geom_rectangle(fltk=False)
    write_xdmf_mesh("rijke", 2)

# Read mesh 
geometry = XDMFReader("MeshDir/rijke")
mesh, cell_tags, facet_tags = geometry.getAll()

# Define the boundary conditions
import params
boundary_conditions = {4: {'Neumann'},
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Define Speed of sound
# c = dolfinx.Constant(mesh, PETSc.ScalarType(1))
c = params.c(mesh)
deg = 1

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree =deg)

matrices.assemble_A()
matrices.assemble_C()

A = matrices.A
C = matrices.C

eigensolver = eps_solver(A,C,np.pi,1,print_results=True)

omega, p = normalize_eigenvector(mesh, eigensolver, 0)

xdmf_writer("Results/p", mesh, p)

