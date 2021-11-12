import dolfinx 
from dolfinx.fem.assemble import assemble_scalar
import numpy as np
import scipy.linalg
import ufl
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
# Take the directional derivative of f in the facet normal direction, Dn(f) := dot(grad(f), n).
from ufl import  FacetNormal, grad, dot, inner
from ufl.operators import Dn 



def _shape_gradient_Dirichlet(c, p_dir, p_adj):
    # Equation 4.34 in thesis
    return - c**2 * Dn(conjugate_function(p_adj)) * Dn (p_dir)


def _shape_gradient_Neumann(c, omega, p_dir, p_adj):
    # Equation 4.35 in thesis
    p_adj_conj = conjugate_function(p_adj)
    return c**2 * dot(grad(p_adj_conj), grad(p_dir)) - omega**2 * p_dir * p_adj_conj



def _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, index):

    # Equation 4.33 in thesis
    curvature = geometry.get_curvature_field(index)
    G = -conjugate_function(p_adj) * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
        _shape_gradient_Neumann(c, omega, p_dir, p_adj) + \
         2 * _shape_gradient_Dirichlet(c, p_dir, p_adj)

    return G

# ________________________________________________________________________________

def ShapeDerivativesParametric(geometry, boundary_conditions, omega, p_dir, p_adj, c, local=False):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = ufl.Measure('ds', domain = mesh, subdomain_data = facet_tags)

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

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags
    
    n = FacetNormal(mesh)
    ds = ufl.Measure('ds', domain = mesh, subdomain_data = facet_tags)

    Q = dolfinx.FunctionSpace(geometry.mesh, ("CG", 1))
    C = dolfinx.Constant(geometry.mesh,1)
    C = dolfinx.interpolate(C, Q)

    for i, value in boundary_conditions.items():
        A = assemble_scalar(C * ds(i))
        C = 1 / A
        ## BELOW THIS LINE IS NOT IMPLEMENTED YET!
        if value == {'Dirichlet'}:
            #DO DEGENERACY
            G = _shape_gradient_Dirichlet(c, p_dir1, p_adj1)
        elif value == {'Neumann'}:
            #DO DEGENERACY
            G = _shape_gradient_Neumann(c, omega, p_dir1, p_adj1)
        else :
            #DO DEGENERACY
            G = _shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj1, i)

        if len(G) == 4:
                    # the eigenvalues are 2-fold degenerate
                    A = np.array(([G[0], G[1]],
                                [G[2], G[3]]))
                    eig = scipy.linalg.eigvals(A)
                    derivatives[j] = eig.tolist()
    



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



