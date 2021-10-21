import dolfinx
import dolfinx.geometry
from dolfinx import FunctionSpace, UnitIntervalMesh
from dolfinx.fem.assemble import assemble_matrix
import basix
import numpy as np
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, inner, Measure,grad
from petsc4py import PETSc
# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 30
mesh = dolfinx.UnitIntervalMesh(MPI.COMM_WORLD, n_elem, dolfinx.cpp.mesh.GhostMode.shared_facet)
# mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n_elem, n_elem)

V = FunctionSpace(mesh, ("Lagrange", degree))
def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
tdim = mesh.topology.dim
marked_cells = dolfinx.mesh.locate_entities(mesh, tdim, fl_subdomain_func)
fl = 1
subdomain = dolfinx.MeshTags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))
dx = Measure("dx", subdomain_data=subdomain)
u = TrialFunction(V) 
v = TestFunction(V)
V_fl = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.Constant(mesh, PETSc.ScalarType(1))*dx(fl)), op=MPI.SUM)
b = dolfinx.Function(V)
b.x.array[:] = 0
a = dolfinx.fem.assemble_vector(b.vector, inner(dolfinx.Constant(mesh, (1/V_fl)), v)*dx(fl))
b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
indices1 = np.array(np.flatnonzero(a.getArray()),dtype=np.int32)
print("INDICES:", indices1, type(indices1))
a = b.x.array
dofmaps = V.dofmap
indices1 = dofmaps.index_map.local_to_global(indices1)
a = list(zip(indices1, a[indices1]))

# VECTOR B

point = np.array([[0.2, 0, 0]])
v = np.array([[0, 0, 1]]).T
if tdim == 1:
    v = np.array([[1]])
elif tdim == 2:
    v = np.array([[1, 0]]).T

# Finds the basis function's derivative at point x
# and returns the relevant dof and derivative as a list
num_local_cells = mesh.topology.index_map(tdim).size_local
bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, np.arange(num_local_cells, dtype=np.int32))
cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point[0])
# Choose one of the cells that contains the point
cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point[0], 1)

# Data required for pull back of coordinate
gdim = mesh.geometry.dim
num_local_cells = mesh.topology.index_map(tdim).size_local
num_dofs_x = mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh
t_imap = mesh.topology.index_map(tdim)
num_cells = t_imap.size_local + t_imap.num_ghosts
x = mesh.geometry.x
x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)
cell_geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
points_ref = np.zeros((1, tdim))

# Data required for evaluation of derivative
ct = dolfinx.cpp.mesh.to_string(mesh.topology.cell_type)
element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), degree, basix.LagrangeVariant.equispaced)
dofmaps = V.dofmap
coordinate_element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), 1, basix.LagrangeVariant.equispaced)

point_ref = None
B = []
if len(cell) > 0:
    cell = cell[0]
    # Only add contribution if cell is owned
    if cell < num_local_cells:
        # Map point in cell back to reference element
        cell_geometry[:] = x[x_dofs[cell], :gdim]
        point_ref = mesh.geometry.cmap.pull_back(point[:,:gdim], cell_geometry)
        dphi = coordinate_element.tabulate(1, point_ref)[1:,0,:]
        J = np.dot(cell_geometry.T, dphi.T)
        Jinv = np.linalg.inv(J)  

        cell_dofs = dofmaps.cell_dofs(cell)
        global_dofs = dofmaps.index_map.local_to_global(cell_dofs)
        # Compute gradient on physical element by multiplying by J^(-T)
        d_dx = (Jinv.T @ element.tabulate(1, point_ref)[1:, 0, :]).T
        d_dv = np.dot(d_dx, v)[:, 0]
        for i in range(len(cell_dofs)):
            B.append([global_dofs[i], d_dv[i]])
    else:
        print(MPI.COMM_WORLD.rank, "Ghost", cell)

root = -1
if len(B) > 0:
    root = MPI.COMM_WORLD.rank
b_root = MPI.COMM_WORLD.allreduce(root, op=MPI.MAX)
# print("PRE", MPI.COMM_WORLD.rank, B, b_root)
B = MPI.COMM_WORLD.bcast(B, root=b_root)
# print("POST", MPI.COMM_WORLD.rank, B, b_root)

print("A: ",a)
print("B: ",B)
def csr_matrix(a, b):
        """RETURNS ROWS, COLUMNS and VALUES for PETSc matrix"""

        nnz = len(a) * len(b)
        

        row = np.zeros(nnz)
        col = np.zeros(nnz)
        val = np.zeros(nnz, dtype=np.complex128)

        for i, c in enumerate(a):
            for j, d in enumerate(b):
                row[i * len(b) + j] = c[0]
                col[i * len(b) + j] = d[0]
                val[i * len(b) + j] = c[1] * d[1]

        row = row.astype(dtype='int32')
        col = col.astype(dtype='int32')
        # print("ROW: ",row,
        # "COL: ",col,
        # "VAL: ",val)
        return row, col, val

def assemble_submatrices(a,b):

    global_size = V.dofmap.index_map.size_global
    local_size = V.dofmap.index_map.size_local
    # print("LOCAL SIZE: ",local_size)

    row, col, val = csr_matrix(a, b)
    
    mat = PETSc.Mat().create(comm = PETSc.COMM_WORLD) 
    mat.setSizes([(local_size, global_size), (local_size, global_size)])
    mat.setType('aij') 
    mat.setUp()
    
    for i in range(len(row)):
        
        mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
        # print(mat.getValue(row[i],col[i]))
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat

D = assemble_submatrices(a,B)

from slepc4py import SLEPc
u = TrialFunction(V) 
v = TestFunction(V)
A = assemble_matrix(inner(grad(u), grad(v))*dx)
A.assemble()

mat = A-D
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setOperators(mat)
eigensolver.setWhichEigenpairs(eigensolver.Which.LARGEST_REAL)

eigensolver.solve()
self = eigensolver.getConverged()
vr, vi = mat.createVecs()
if self > 0:
    for i in range (self):
            l = eigensolver.getEigenpair(i ,vr, vi)
            print(l.real)

            