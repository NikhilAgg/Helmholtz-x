from dolfinx.fem import FunctionSpace,Constant
from dolfinx.mesh import meshtags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlameNT
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
import params_dim


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 1000
mesh = create_unit_interval(MPI.COMM_WORLD, n_elem)
V = FunctionSpace(mesh, ("Lagrange", degree))

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
tdim = mesh.topology.dim
marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define the boundary conditions

# boundary_conditions = {1: {'Robin': params_dim.Y_in},  # inlet
#                        2: {'Robin': params_dim.Y_out}}  # outlet

boundary_conditions = {1: {'Dirichlet'},  # inlet
                       2: {'Dirichlet'}}  # outlet

# Define Speed of sound

# c = params_dim.c(mesh)
from petsc4py import PETSc
c = Constant(mesh, PETSc.ScalarType(400))
# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C



# x_f = params_dim.x_f
# x_r = params_dim.x_r

# n = params_dim.n(mesh, x_f[0][0])
n = params_dim.n_1D
tau = params_dim.tau_linear(mesh,params_dim.x_f, params_dim.a_f, params_dim.tau_u, params_dim.tau_d, degree=degree)

rho = params_dim.rho(mesh, params_dim.x_f, params_dim.a_f, params_dim.rho_d, params_dim.rho_u)
w = params_dim.w(mesh, params_dim.x_r, params_dim.a_r)
h = params_dim.h(mesh, params_dim.x_f, params_dim.a_f)


target = 200 * 2 * np.pi # 150 * 2 * np.pi

D = ActiveFlameNT(mesh, subdomains, w, h, rho, params_dim.Q_tot, params_dim.U_bulk, n, tau,  
                    degree=degree)

E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

if MPI.COMM_WORLD.rank == 0:
    print(f"Eigenfrequency ->  {omega/(2*np.pi):.3f}")

plt.plot(uh.x.array.real)
plt.savefig("Results/1Dactive_real.png")
plt.clf()

plt.plot(uh.x.array.imag)
plt.savefig("Results/1Dactive_imag.png")
