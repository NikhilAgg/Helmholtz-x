import datetime
start_time = datetime.datetime.now()
from dolfinx.fem import Constant
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_eps
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlameNT
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.parameters_utils import rho,tau_linear,h,w
from helmholtz_x.helmholtz_pkgx.dolfinx_utils import xdmf_writer
from helmholtz_x.geometry_pkgx.xdmf_utils import XDMFReader
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import params

rijke2d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke2d.getAll()
rijke2d.getInfo()

boundary_conditions = {4: {'Neumann'},
                       3: {'Dirichlet'},
                       2: {'Neumann'},
                       1: {'Dirichlet'}}

degree = 1

c = Constant(mesh, PETSc.ScalarType(400))

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c , degree = degree)

matrices.assemble_A()
matrices.assemble_C()

A = matrices.A
C = matrices.C

rho = rho(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = w(mesh, params.x_r, params.a_r)
h = h(mesh, params.x_f, params.a_f)

n = params.n_2D
tau = tau_linear(mesh,params.x_f, params.a_f, params.tau_u, params.tau_d, degree=degree)

target = 200 * 2 * np.pi

D = ActiveFlameNT(mesh, subdomains, w, h, rho, params.Q_tot, params.U_bulk, n, tau, 
                    degree=degree)
                    
E = fixed_point_iteration_eps(matrices, D, target**2, nev=2, i=0, print_results= False)
omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, p)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
