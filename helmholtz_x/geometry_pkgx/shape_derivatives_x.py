

from dolfinx.fem import Constant, VectorFunctionSpace, Function, DirichletBC, locate_dofs_topological, set_bc
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, dot, inner, Measure
from ufl.operators import Dn, facet_avg #Dn(f) := dot(grad(f), n).
from petsc4py import PETSc
from mpi4py import MPI
from geomdl import BSpline, utilities, helpers

import numpy as np
import scipy.linalg
import os



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

def ShapeDerivativesParametric(geometry, boundary_conditions, omega, p_dir, p_adj, c):

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


def ShapeDerivatives3DRijke(geometry, boundary_conditions, omega, p_dir, p_adj, c, boundary_index, control_points):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    if boundary_conditions[boundary_index] == 'Dirichlet':
        G = _shape_gradient_Dirichlet(c, p_dir, p_adj)
    elif boundary_conditions[boundary_index] == 'Neumann':
        G = _shape_gradient_Neumann(c, omega, p_dir, p_adj)
        print("NEUMANN WORKED")
    elif boundary_conditions[boundary_index]['Robin'] :
        print("ROBIN WORKED")
        G = _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj, boundary_index)
            
    Field = _displacement_field(geometry, control_points, boundary_index)

    derivatives = np.zeros((len(Field)), dtype=complex)

    for control_point_index, V in enumerate(Field):

        derivatives[control_point_index] = assemble_scalar( inner(V, n) * G * ds(boundary_index) )
            
    return derivatives

def _displacement_field(geometry,  points, boundary_index):
    """ This function calculates displacement field as dolfinx function.
        It only works for cylindical geometry with length L.

    Args:
 
        mesh ([dolfinx.mesh.Mesh ]): mesh
        points ([int]): Control points of geometry
        
    Returns:
        list of Displacement Field function for each control point [dolfinx.fem.function.Function]
    """

    # Fix file path
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Create a B-Spline curve instance
    curve = BSpline.Curve()

    # Set up curve
    curve.degree = 3
    curve.ctrlpts = points

    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.001

    # Evaluate curve
    curve.evaluate()

    DisplacementField = [None] * len(points)

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags
    
    for control_point_index in range(len(points)):

        u = np.linspace(min(curve.knotvector),max(curve.knotvector),len(curve.evalpts))
        V = [helpers.basis_function_one(curve.degree, curve.knotvector, control_point_index, i) for i in u]

        Q = VectorFunctionSpace(mesh, ("CG", 1))

        gdim = mesh.topology.dim

        def V_function(x):
            scaler = points[-1][2] # THIS MIGHT NEEDS TO BE FIXED.. Cylinder's control points should be starting from 0 to L on z-axis.
            V_poly = np.poly1d(np.polyfit(u*scaler, np.array(V), 10))
            theta = np.arctan2(x[1],x[0])  
            values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
            values[0] = V_poly(x[2])*np.cos(theta)
            values[1] = V_poly(x[2])*np.sin(theta)
            return values

        temp = Function(Q)
        temp.interpolate(V_function)
        temp.name = 'V'

        facets = facet_tags.indices[facet_tags.values == boundary_index]
        dbc = DirichletBC(temp, locate_dofs_topological(Q, gdim-1, facets))

        DisplacementField[control_point_index] = Function(Q)
        DisplacementField[control_point_index].vector.set(0)

        set_bc(DisplacementField[control_point_index].vector,[dbc])

    return DisplacementField