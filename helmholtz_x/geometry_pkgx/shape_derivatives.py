import dolfinx
from helmholtz_x.helmholtz_pkgx.petsc4py_utils import conjugate_function
from dolfinx.fem.assemble import assemble_scalar
import numpy as np
from ufl import  FacetNormal, grad, dot, div, inner, Measure, conditional, conj
from ufl.operators import Dn 
from mpi4py import MPI

class ShapeDerivatives:

    def __init__(self, geometry, boundary_conditions, p_dir, p_adj_normalized, c):
        """
        Shape derivative class for geometry which has different BCs

        Parameters
        ----------
        geometry : Geometry
            Geometry object produced by this module
        boundary_conditions : dict
            Boundary conditions dictionary
        p_dir : Coefficient
            Direct pressure eigenfunction 
        p_adj : Coefficient
            NORMALIZED Adjoint pressure eigenfunction 
        c : Coefficient
            Speed of sound, it should be interpolated on defined function space!

        """

        self.c = c
        
        self.p_dir = p_dir
        self.p_adj = p_adj_normalized
        self.p_adj_conj = conjugate_function(p_adj_normalized)
        self.boundary_conditions = boundary_conditions

        self.geometry = geometry

        self.mesh = self.geometry.mesh
        self.facet_tags = self.geometry.facet_tags
        self.n = FacetNormal(self.mesh)
        self.ds =  Measure('ds', domain=self.mesh, subdomain_data=self.facet_tags)


    def integral_1(self):

        c = self.c
        form = -c**2 * Dn(self.p_adj_conj) * Dn(self.p_dir)

        return form

    def integral_2(self):

        c = self.c

        form = c**2 * self.p_adj_conj * Dn(Dn(self.p_dir))

        return form

    def common_term(self):

        c = self.c
        n = self.n

        form = c**2 * (self.p_adj_conj)*( grad(self.p_dir) - Dn(self.p_dir)*n)

        return form

    def integral_3(self):

        curvature = self.curvature
        n = self.n

        common_term = self.common_term()

        form =  curvature * inner(common_term, n) + div( common_term) - inner(dot(grad(common_term), n), n)

        return form
    #

    def get_Dirichlet(self):

        return self.integral_1()

    def get_Neumann(self):

        return self.integral_2() + self.integral_3()

    def get_Robin(self):

        return self.integral_1()+ self.integral_2() + self.integral_3()

    def __call__(self, i):

        n = self.n
        ds = self.ds

        ctrl_pts = self.geometry.ctrl_pts[i]  # ndarray
        
        self.curvature = self.geometry.get_curvature_field(i)

        shape_derivatives = np.zeros((len(ctrl_pts), 2), dtype=complex)

        for j in range(len(ctrl_pts)):
            
            V_x, V_y = self.geometry.get_displacement_field(i, j) 

            if "Dirichlet" in self.boundary_conditions[i]:

                G = self.get_Dirichlet()
                # print(assemble_scalar(G_dir*ds(i)))

                # gradient_x = assemble_scalar( inner(V_x, n) * G_dir * ds(i) )
                # gradient_y = assemble_scalar( inner(V_y, n) * G_dir * ds(i) )

            elif "Neumann" in self.boundary_conditions[i]:

                G = self.get_Neumann()

                # gradient_x = assemble_scalar( inner(V_x, n) * G_neu * ds(i) )
                # gradient_y = assemble_scalar( inner(V_y, n) * G_neu * ds(i) )

            elif "Robin" in self.boundary_conditions[i]:

                G = self.get_Robin()

            gradient_x = assemble_scalar( inner(V_x, n) * G * ds(i) )
            gradient_y = assemble_scalar( inner(V_y, n) * G * ds(i) )

            MPI.COMM_WORLD.allreduce(gradient_x , op=MPI.SUM) # For parallel executions
            MPI.COMM_WORLD.allreduce(gradient_y , op=MPI.SUM) # For parallel executions

            shape_derivatives[j][0] = gradient_x
            shape_derivatives[j][1] = gradient_y

        return shape_derivatives
