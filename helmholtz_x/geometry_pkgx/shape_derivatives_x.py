from dolfinx.fem import Constant 
import numpy as np
import scipy.linalg

from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, dot, inner, Measure
from ufl.operators import Dn #Dn(f) := dot(grad(f), n).
from petsc4py import PETSc
from mpi4py import MPI


def _shape_gradient_Dirichlet(c, p_dir, p_adj):
    # Equation 4.34 in thesis
    return - c**2 * Dn(conjugate_function(p_adj)) * Dn (p_dir)


def _shape_gradient_Neumann(c, omega, p_dir, p_adj):
    # Equation 4.35 in thesis
    p_adj_conj = conjugate_function(p_adj)
    return c**2 * dot(grad(p_adj_conj), grad(p_dir)) - omega**2 * p_dir * p_adj_conj



def _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, index):

    # FIX ME FOR PARAMETRIC 3D
    if geometry.mesh.topology.dim == 2:
        curvature = geometry.get_curvature_field(index)
    else:
        curvature = 0

    # Equation 4.33 in thesis
    G = -conjugate_function(p_adj) * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
        _shape_gradient_Neumann(c, omega, p_dir, p_adj) + \
         2 * _shape_gradient_Dirichlet(c, p_dir, p_adj)

    return G

# ________________________________________________________________________________

def ShapeDerivativesParametric(geometry, boundary_conditions, omega, p_dir, p_adj, c, local=False):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    results = {}

    for i, value in boundary_conditions.items():
        
        if i in geometry.ctrl_pts:
            if value == {'Dirichlet'}:
                G = _shape_gradient_Dirichlet(c, p_dir, p_adj)
            elif value == {'Neumann'}:
                G = _shape_gradient_Neumann(c, omega, p_dir, p_adj)
            else :
                G = _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, i)
                
            
            derivatives = np.zeros((len(geometry.ctrl_pts[i]),2), dtype=complex)

            for j in range(len(geometry.ctrl_pts[i])):

                V_x, V_y = geometry.get_displacement_field(i,j)

                derivatives[j][0] = assemble_scalar( inner(V_x, n) * G * ds(i) )
                derivatives[j][1] = assemble_scalar( inner(V_y, n) * G * ds(i) )
                
 
            results[i] = derivatives
            
    return results


def ShapeDerivativesDegenerate(geometry, boundary_conditions, omega, 
                               p_dir1, p_dir2, p_adj1, p_adj2, c):
    
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    
    results = {} 

    for tag, value in boundary_conditions.items():
        C = Constant(geometry.mesh, PETSc.ScalarType(1))
        A = assemble_scalar(C * ds(tag))
        A = geometry.mesh.allreduce(A, op=MPI.SUM) # For parallel runs
        C = 1 / A

        G = []
        if value == {'Dirichlet'}:

            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj1))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj1))
            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj2))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj2))
        elif value == {'Neumann'}:
            
            G.append(_shape_gradient_Neumann(c, omega, p_dir1, p_adj1))
            G.append(_shape_gradient_Neumann(c, omega, p_dir2, p_adj1))
            G.append(_shape_gradient_Neumann(c, omega, p_dir1, p_adj2))
            G.append(_shape_gradient_Neumann(c, omega, p_dir2, p_adj2))
        else :

            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj1, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj1, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj2, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj2, tag))
        
        # the eigenvalues are 2-fold degenerate
        for index,form in enumerate(G):
            G[index] = assemble_scalar(C * form *ds(tag))
        A = np.array(([G[0], G[1]],
                      [G[2], G[3]]))
        
        eig = scipy.linalg.eigvals(A)
        print("eig: ",eig)
        results[tag] = eig.tolist()
    
    return results



if __name__=='__main__':
    lcar =0.2

    # p0 = [0., + .0235]
    # p1 = [0., - .0235]
    # p2 = [1., - .0235]
    # p3 = [1., + .0235]

    p0 = [0., + 0.5]
    p1 = [0., - .5]
    p2 = [1., - .5]
    p3 = [1., + .5]

    points  = [p0, p1, p2, p3]

    edges = {1:{"points":[points[0], points[1]], "parametrization": False},
             2:{"points":[points[1], points[2]], "parametrization": True, "numctrlpoints":3},
             3:{"points":[points[2], points[3]], "parametrization": False},
             4:{"points":[points[3], points[0]], "parametrization": True, "numctrlpoints":3}}



