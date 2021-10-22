import dolfinx
from dolfinx import Function, FunctionSpace
from dolfinx.fem.assemble import assemble_matrix
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc
import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("geom")

lc = 5e-1
L = 1

geom = gmsh.model.geo

p1 = geom.addPoint(0, 0, 0, lc)
p2 = geom.addPoint(L, 0, 0, lc)
p3 = geom.addPoint(L, L, 0, lc)
p4 = geom.addPoint(0, L, 0, lc)

l1 = geom.addLine(1, 2)
l2 = geom.addLine(2, 3)
l3 = geom.addLine(3, 4)
l4 = geom.addLine(4, 1)
ll1 = geom.addCurveLoop([1, 2, 3, 4])
s1 = geom.addPlaneSurface([1])
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1], 2) # Bottom Wall 
gmsh.model.addPhysicalGroup(1, [2], 3) # Outlet
gmsh.model.addPhysicalGroup(1, [3], 4) # Upper Wall
gmsh.model.addPhysicalGroup(1, [4], 1) # Inlet
gmsh.model.addPhysicalGroup(2, [1], 0)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

gmsh.option.setNumber("Mesh.SaveAll", 0)
gmsh.write("geom.msh")

from gmsh_helpers import read_from_msh
mesh, cell_tags, facet_tags = read_from_msh("geom.msh", cell_data=True, facet_data=True, gdim=2)

V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

a = -inner(grad(u), grad(v))*dx
A = assemble_matrix(a)
A.assemble()

c = inner(u , v) * dx
C = assemble_matrix(c)
C.assemble()

solver = SLEPc.EPS().create(MPI.COMM_WORLD)
C = - C

solver.setOperators(A, C)
solver.setFromOptions()
solver.solve()

A = solver.getOperators()[0]
vr, vi = A.createVecs()

eig = solver.getEigenvalue(0)
omega = np.sqrt(eig)
solver.getEigenvector(0, vr, vi)
print(omega)

p = Function(V)
p.vector.setArray(vr.array)
import dolfinx.io
p.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)
