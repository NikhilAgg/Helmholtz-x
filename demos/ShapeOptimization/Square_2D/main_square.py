
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import pep_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector, normalize_adjoint
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import vector_matrix_vector
from helmholtz_x.geometry_pkgx.geometry import Geometry
from math import pi

import dolfinx
import os
import matplotlib.pyplot as plt
import params

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower

if not os.path.exists("MeshDir"):
    os.mkdir("MeshDir")
    
lcar = 0.02

p0 = [0., + .5]
p1 = [0., - .5]
p2 = [1., - .5]
p3 = [1., + .5]

points  = [p0, p1, p2, p3]

edges = {1:{"points":[points[0], points[1]], "parametrization": False},
         2:{"points":[points[1], points[2]], "parametrization": True, "numctrlpoints":6},
         3:{"points":[points[2], points[3]], "parametrization": False},
         4:{"points":[points[3], points[0]], "parametrization": True, "numctrlpoints":6}}



geometry = Geometry("MeshDir/ekrem", points, edges, lcar)
geometry.make_mesh(False)

boundary_conditions = {4: {'Neumann'},
                       3: {'Robin': params.Y_out},
                       2: {'Neumann'},
                       1: {'Robin': params.Y_in}}


degree = 2

c = params.c(geometry.mesh)

matrices = PassiveFlame(geometry.mesh, geometry.facet_tags, boundary_conditions, c , degree = degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

E = pep_solver(matrices.A, matrices.B, matrices.C, pi, nev=2, print_results = False)
E_adj = pep_solver(matrices.A, matrices.B_adj, matrices.C, pi, nev=2, print_results = False)

omega_dir, p_dir = normalize_eigenvector(geometry.mesh, E, i=0, degree=degree)

omega_adj, p_adj = normalize_eigenvector(geometry.mesh, E_adj, i=0, degree=degree)

print("OMEGA_DIR: ", omega_dir)
print("OMEGA_ADJ: ", omega_adj)

p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, matrices)

from helmholtz_x.geometry_pkgx.shape_derivatives import ShapeDerivatives

shapeder = ShapeDerivatives(geometry, boundary_conditions, p_dir, p_adj_norm, c)

shape_der_2 = shapeder(2)

print(shape_der_2)




