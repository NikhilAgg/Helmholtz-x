import dolfinx 
from dolfinx.fem.assemble import assemble_scalar
import numpy as np
import scipy.linalg
import ufl
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
# Take the directional derivative of f in the facet normal direction, Dn(f) := dot(grad(f), n).
from ufl import  FacetNormal
from ufl.operators import Dn 
grad = ufl.grad
dot = ufl.dot
div = ufl.div
inner = ufl.inner
# ________________________________________________________________________________



def _shape_gradient_Dirichlet(c, p_dir, p_adj):
    
    return - c**2 * inner(grad(conjugate_function(p_adj)), grad(p_dir))


def _shape_gradient_Neumann(c, omega, p_dir, p_adj):
    
    return c**2 * inner(grad(conjugate_function(p_adj)), grad(p_dir)) - omega**2 * conjugate_function(p_adj) * p_dir



def _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, index):
 
    # V = dolfinx.FunctionSpace(mesh, 'CG', 1)
    # c = dolfinx.interpolate(c, V)

    # Equation 4.33 in thesis
    curvature = geometry.get_curvature_field(index)
    G = -conjugate_function(p_adj) * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
        _shape_gradient_Neumann(c, omega, p_dir, p_adj) + \
            2 * _shape_gradient_Dirichlet(c, p_dir, p_adj)

    return G

# ________________________________________________________________________________

def shape_derivatives(geometry, boundary_conditions, omega, p_dir, p_adj, c, local=False):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = ufl.Measure('ds', subdomain_data=facet_tags)

    G_Dir = []
    G_Neu = []
    G_Rob = []

    results = {}

    # shape gradient, ufl forms for Dirichlet and Neumann boundary conditions

    if 'Dirichlet' in boundary_conditions.values():
        
        G_Dir.append(_shape_gradient_Dirichlet(c, p_dir, p_adj))

    if 'Neumann' in boundary_conditions.values():
        G_Neu.append(_shape_gradient_Neumann(c, omega, p_dir, p_adj))

    

    for i, value in boundary_conditions.items():
        
        if i in geometry.ctrl_pts:
            if value == 'Dirichlet':
                G = G_Dir
            elif value == 'Neumann':
                G = G_Neu
            # elif list(value.keys())[0] == 'Robin':
            else :
                G_Rob.append(_shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, i))
                G = G_Rob
            
            
            
            derivatives = np.zeros((len(geometry.ctrl_pts[i])), dtype=complex)
            # print("LENGTH OF CONTROL POINTS: ", len(geometry.ctrl_pts[i]))
            for j in range(len(geometry.ctrl_pts[i])):

                V = geometry.get_displacement_field(i,j)

                C = inner(V, n) # Displacements only y direction
                
                g = assemble_scalar(inner(C, G[0]) * ds(i)) # assemble_scalar(C * G[0] * ds(i)) gives different sign for imag part
                
                    # print("g is: ",g[0], )
                if type(g) == complex:
                    # the eigenvalue is simple
                    derivatives[j] = g
                elif len(g) == 4:
                    # the eigenvalues are 2-fold degenerate
                    A = np.array(([g[0], g[1]],
                                [g[2], g[3]]))
                    eig = scipy.linalg.eigvals(A)
                    derivatives[j] = eig.tolist()
                results[i] = derivatives
            
    return results

def shape_dirichlet(geometry, boundary_conditions, omega, p_dir, p_adj, c, index):
    
    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = ufl.Measure('ds', subdomain_data=facet_tags)
    
    G = _shape_gradient_Dirichlet(c, p_dir, p_adj)

    results = {}

    derivatives = np.zeros((len(geometry.ctrl_pts[index])), dtype=complex)
    
    for j in range(len(geometry.ctrl_pts[index])):

            V = geometry.get_displacement_field(index,j)

            C = inner(V, n) # Displacements only y direction
            
            g = assemble_scalar(inner(C, G) * ds(index))
            
                # print("g is: ",g[0], )
            if type(g) == complex:
                # the eigenvalue is simple
                derivatives[j] = g
            elif len(g) == 4:
                # the eigenvalues are 2-fold degenerate
                A = np.array(([g[0], g[1]],
                            [g[2], g[3]]))
                eig = scipy.linalg.eigvals(A)
                derivatives[j] = eig.tolist()
            results[index] = derivatives
    return results

def shape_neumann(geometry, boundary_conditions, omega, p_dir, p_adj, c, index):
    
    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = ufl.Measure('ds', subdomain_data=facet_tags)
    
    G = _shape_gradient_Neumann(c, omega, p_dir, p_adj)

    results = {}

    derivatives = np.zeros((len(geometry.ctrl_pts[index])), dtype=complex)
    
    for j in range(len(geometry.ctrl_pts[index])):

            V = geometry.get_displacement_field(index,j)

            C = inner(V, n) # Displacements only y direction
            
            g = assemble_scalar(inner(C, G) * ds(index))
            
                # print("g is: ",g[0], )
            if type(g) == complex:
                # the eigenvalue is simple
                derivatives[j] = g
            elif len(g) == 4:
                # the eigenvalues are 2-fold degenerate
                A = np.array(([g[0], g[1]],
                            [g[2], g[3]]))
                eig = scipy.linalg.eigvals(A)
                derivatives[j] = eig.tolist()
            results[index] = derivatives
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



