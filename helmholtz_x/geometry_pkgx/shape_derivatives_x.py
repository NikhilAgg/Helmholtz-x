

from dolfinx.fem import Constant, VectorFunctionSpace, Function, dirichletbc, locate_dofs_topological, set_bc, form
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, dot, inner, Measure, div, variable
from ufl.operators import Dn #Dn(f) := dot(grad(f), n).
from petsc4py import PETSc
from mpi4py import MPI
from geomdl import BSpline, utilities, helpers

import numpy as np
import scipy.linalg
import os



def _shape_gradient_Dirichlet(c, p_dir, p_adj_conj):
    # Equation 4.34 in thesis
    return - c**2 * Dn(p_adj_conj) * Dn (p_dir)

def _shape_gradient_Neumann(c, p_dir, p_adj_conj):
    # Equation 4.35 in thesis
    # p_adj_conj = conjugate_function(p_adj)
    return  div(p_adj_conj * c**2 * grad(p_dir))

def _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj):
    # Equation 4.36 in thesis
    return c**2 * dot(grad(p_adj_conj), grad(p_dir)) - omega**2 * p_adj_conj * p_dir  

def _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj_conj, index):

    # FIX ME FOR PARAMETRIC 3D
    if geometry.mesh.topology.dim == 2:
        curvature = geometry.get_curvature_field(index)
    else:
        curvature = 0

    # Equation 4.33 in thesis
    G = -p_adj_conj * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
        _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj) + \
         2 * _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)

    return G

# ________________________________________________________________________________

def ShapeDerivativesParametric2D(geometry, boundary_conditions, omega, p_dir, p_adj, c):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags
    
    n = FacetNormal(mesh)
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    p_adj_conj = conjugate_function(p_adj) # it should be conjugated once

    results = {}
    
    for tag, value in boundary_conditions.items():
        
        if tag in geometry.ctrl_pts:
            derivatives = np.zeros((len(geometry.ctrl_pts[tag]),2), dtype=complex)

            if value == 'Dirichlet':
                G = _shape_gradient_Dirichlet(c, p_dir, p_adj)
            elif value == 'Neumann':
                G = _shape_gradient_Neumann(c, p_dir, p_adj_conj)
            else :
                G = _shape_gradient_Robin(geometry, c, omega, p_dir,  p_adj_conj, tag)

            for j in range(len(geometry.ctrl_pts[tag])):

                V_x, V_y = geometry.get_displacement_field(tag,j)
                
                derivatives[j][0] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_x, n) * G * ds(tag))), op=MPI.SUM)
                derivatives[j][1] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_y, n) * G * ds(tag))), op=MPI.SUM)
            
            results[tag] = derivatives
            
    return results

def ShapeDerivativesLocal2D(geometry, boundary_conditions, omega, p_dir, p_adj, c):

    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)

    p_adj_conj = conjugate_function(p_adj) # it should be conjugated once

    results = {}

    const = Constant(geometry.mesh, PETSc.ScalarType(1))

    for tag, value in boundary_conditions.items():
        
        L = MPI.COMM_WORLD.allreduce(assemble_scalar(const * ds(tag)), op=MPI.SUM)
        # print("L: ", L)
        C = const / L
        
        if value == 'Dirichlet':
            G = _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
        elif value == 'Neumann':
            G = _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj)
        else :
            curvature = 0
            G = -p_adj_conj * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
            _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj) + \
            2 * _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
        print("CHECK: ", MPI.COMM_WORLD.allreduce(assemble_scalar(G * ds(tag)), op=MPI.SUM)) 
        results[tag] = MPI.COMM_WORLD.allreduce(assemble_scalar(C * G * ds(tag)), op=MPI.SUM)
            
    return results

class ShapeDerivativesRijke2D:

    def __init__(self,geometry, boundary_conditions, omega, p_dir, p_adj, c):
        
        self.geometry = geometry
        self.mesh = geometry.mesh    
        self.facet_tags = geometry.facet_tags
        self.c = c
        self.n = FacetNormal(self.mesh)
        self.ds = Measure('ds', domain = self.mesh, subdomain_data = self.facet_tags)

        self.boundary_conditions = boundary_conditions
        self.omega = omega

        self.p_dir = p_dir
        self.p_adj = p_adj
        self.p_adj_conj = conjugate_function(p_adj)
        self.results = {}
        

    def _shape_gradient_Dirichlet(self):
        # Equation 4.34 in thesis
        return - self.c**2 * Dn(self.p_adj_conj) * Dn (self.p_dir)

    def _shape_gradient_Neumann(self):
        # Equation 4.35 in thesis
        return  div(self.p_adj_conj * self.c**2 * grad(self.p_dir))

    def _shape_gradient_Neumann2(self):
        # Equation 4.36 in thesis
        return self.c**2 * dot(grad(self.p_adj_conj), grad(self.p_dir)) - self.omega**2 * self.p_adj_conj * self.p_dir  

    def _shape_gradient_Robin(self, index):

        # FIX ME FOR PARAMETRIC 3D
        if self.geometry.mesh.topology.dim == 2:
            curvature = self.geometry.get_curvature_field(index)
        else:
            curvature = 0

        # Equation 4.33 in thesis
        return -self.p_adj_conj * (curvature * self.c ** 2 + self.c * Dn(self.c)*Dn(self.p_dir)) + \
            self._shape_gradient_Neumann() + \
            2 * self._shape_gradient_Dirichlet()

    def calculate(self):

        boundary_conditions = self.boundary_conditions
        geometry = self.geometry
        n = self.n
        ds = self.ds

        for tag, value in boundary_conditions.items():

            if tag in geometry.ctrl_pts:
                
                derivatives = np.zeros((len(geometry.ctrl_pts[tag]),2), dtype=complex)

                for j in range(len(geometry.ctrl_pts[tag])):
            
                    V_x, V_y = self.geometry.get_displacement_field(tag, j) 

                    if value ==  "Dirichlet" :

                        G_dir = self._shape_gradient_Dirichlet()
                        # print(assemble_scalar(G_dir*ds(i)))

                        gradient_x = assemble_scalar( inner(V_x, n) * G_dir * ds(tag) )
                        gradient_y = assemble_scalar( inner(V_y, n) * G_dir * ds(tag) )

                    elif value == "Neumann":

                        G_neu = self._shape_gradient_Neumann()

                        gradient_x = assemble_scalar( inner(V_x, n) * G_neu * ds(tag) )
                        gradient_y = assemble_scalar( inner(V_y, n) * G_neu * ds(tag) )

                    elif "Robin" in self.boundary_conditions[tag]:

                        G_rob = self.get_Robin()

                        gradient_x = assemble_scalar( inner(V_x, n) * G_rob * ds(tag) )
                        gradient_y = assemble_scalar( inner(V_y, n) * G_rob * ds(tag) )

                    # MPI.COMM_WORLD.allreduce(gradient_x , op=MPI.SUM) # For parallel executions
                    # MPI.COMM_WORLD.allreduce(gradient_y , op=MPI.SUM) # For parallel executions

                    derivatives[j][0] = gradient_x
                    derivatives[j][1] = gradient_y
                    
                self.results[tag] = derivatives
    

def ShapeDerivativesDegenerate(geometry, boundary_conditions, omega, 
                               p_dir1, p_dir2, p_adj1, p_adj2, c):
    
    #Clean Master and Slave tags
    for k in list(boundary_conditions.keys()):
        if boundary_conditions[k]=='Master' or boundary_conditions[k]=='Slave':
            del boundary_conditions[k]

    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    p_adj1_conj, p_adj2_conj = conjugate_function(p_adj1), conjugate_function(p_adj2)
    results = {} 

    for tag, value in boundary_conditions.items():
        C = Constant(geometry.mesh, PETSc.ScalarType(1))
        A = assemble_scalar(form(C * ds(tag)))
        A = MPI.COMM_WORLD.allreduce(A, op=MPI.SUM) # For parallel runs
        C = C / A

        G = []
        if value == 'Dirichlet':

            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj1_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj1_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj2_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj2_conj))
        elif value == 'Neumann':
            
            G.append(_shape_gradient_Neumann2(c, omega, p_dir1, p_adj1_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir2, p_adj1_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir1, p_adj2_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir2, p_adj2_conj))
        else :

            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj1_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj1_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj2_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj2_conj, tag))
        
        # the eigenvalues are 2-fold degenerate
        for index,uflform in enumerate(G):
            # value = assemble_scalar(C * form *ds(tag))
            G[index] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(C * uflform *ds(tag))), op=MPI.SUM)
        A = np.array(([G[0], G[1]],
                      [G[2], G[3]]))
        
        eig = scipy.linalg.eigvals(A)
        # print("eig: ",eig)
        results[tag] = eig.tolist()
    
    return results


def ShapeDerivatives3DRijke(geometry, boundary_conditions, omega, p_dir, p_adj, c, boundary_index, control_points):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags

    n = FacetNormal(mesh)
    
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    p_adj_conj = conjugate_function(p_adj) # it should be conjugated once

    if boundary_conditions[boundary_index] == 'Dirichlet':
        G = _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
    elif boundary_conditions[boundary_index] == 'Neumann':
        G = _shape_gradient_Neumann(c, p_dir, p_adj_conj)
        print("NEUMANN WORKED")
    elif boundary_conditions[boundary_index]['Robin'] :
        print("ROBIN WORKED")
        G = _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj_conj, boundary_index)
            
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
        dbc = dirichletbc(temp, locate_dofs_topological(Q, gdim-1, facets))

        DisplacementField[control_point_index] = Function(Q)
        DisplacementField[control_point_index].vector.set(0)

        set_bc(DisplacementField[control_point_index].vector,[dbc])

    return DisplacementField