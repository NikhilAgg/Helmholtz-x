from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological, Constant
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem.assemble import assemble_scalar
from .solver_utils import info
from .parameters_utils import sound_speed_variable_gamma, gamma_function
from ufl import Measure, TestFunction, TrialFunction, grad, inner
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

class PassiveFlame:

    def __init__(self, mesh, facet_tags, boundary_conditions,
                 parameter, degree=1):
        """

        This class defines the matrices A,B,C in
        A + wB + w^2 C = 0 for the passive/active flame problem that consists Robin BC.
    
        Parameters
        ----------
        mesh : dolfinx.cpp.mesh.Mesh
            mesh of the domain
        facet_tags : dolfinx.cpp.mesh.MeshFunctionSizet
            boundary data of the mesh
        boundary_conditions : dict
            boundary conditions for corresponding mesh.
        c : dolfinx.fem.function.Function
            Speed of sound
        degree : int, optional
            degree of the basis functions. The default is 1.
    
        Returns
        -------
        A : petsc4py.PETSc.Mat
            Matrix of Grad term
        B : petsc4py.PETSc.Mat
            Matrix for Robin boundary condition
        C : petsc4py.PETSc.Mat
            Matrix of w^2 term.
        """

        self.mesh = mesh
        self.facet_tag = facet_tags
        self.fdim = mesh.topology.dim - 1
        self.boundary_conditions = boundary_conditions
        self.parameter = parameter
        self.degree = degree

        self.dx = Measure('dx', domain=mesh)
        self.ds = Measure('ds', domain=mesh, subdomain_data=facet_tags)

        self.V = FunctionSpace(mesh, ("Lagrange", degree))

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.AreaConstant = Constant(self.mesh, PETSc.ScalarType(1))

        if self.parameter.name =="temperature":
            self.c = sound_speed_variable_gamma(mesh, parameter)
            self.gamma = gamma_function(self.mesh, self.parameter)
            info("/\ Temperature function is used for passive flame matrices.")
        else:
            self.c = parameter
            info("\/ Speed of sound function is used for passive flame matrices.")

        self.bcs = []
        self.integrals_R = []

        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                u_bc = Function(self.V)
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                dofs = locate_dofs_topological(self.V, self.fdim, facets)
                bc = dirichletbc(u_bc, dofs)
                self.bcs.append(bc)

            if 'Robin' in boundary_conditions[i]:
                R = boundary_conditions[i]['Robin']
                Z = (1+R)/(1-R)
                integrals_Impedance = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integrals_Impedance)   

            if 'ChokedInlet' in boundary_conditions[i]:
                # https://www.oscilos.com/download/OSCILOS_Long_Tech_report.pdf
                A_inlet = assemble_scalar(form(self.AreaConstant * self.ds(i)))
                gamma_inlet_form = form(self.gamma/A_inlet* self.ds(i))
                gamma_inlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_inlet_form), op=MPI.SUM)

                Mach = boundary_conditions[i]['ChokedInlet']
                R = (1-gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))/(1+gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))
                Z = (1+R)/(1-R)
                integral_C_i = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_i) 

            if 'ChokedOutlet' in boundary_conditions[i]:
                # https://www.oscilos.com/download/OSCILOS_Long_Tech_report.pdf
                A_outlet = assemble_scalar(form(self.AreaConstant * self.ds(i)))
                gamma_outlet_form = form(self.gamma/A_outlet* self.ds(i))
                gamma_outlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_outlet_form), op=MPI.SUM)
                
                Mach = boundary_conditions[i]['ChokedOutlet']
                R = (1-0.5*(gamma_outlet-1)*Mach)/(1+0.5*(gamma_outlet-1)*Mach)
                Z = (1+R)/(1-R)
                integral_C_o = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_o) 

        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None

        info("- Passive matrices are assembling..")

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def B_adj(self):
        return self._B_adj

    @property
    def C(self):
        return self._C

    def assemble_A(self):

        a = form(-self.c**2 * inner(grad(self.u), grad(self.v))*self.dx)
        A = assemble_matrix(a, bcs=self.bcs)
        A.assemble()
        info("- Matrix A is assembled.")
        self._A = A

    def assemble_B(self):

        N = self.V.dofmap.index_map.size_global
        n = self.V.dofmap.index_map.size_local

        if self.integrals_R:
            B = assemble_matrix(form(sum(self.integrals_R)))
            B.assemble()
        else:
            B = PETSc.Mat().create()
            B.setSizes([(n, N), (n, N)])
            B.setFromOptions()
            B.setUp()
            B.assemble()
            info("! Note: It can be faster to use EPS solver.")

        B_adj = B.copy()
        B_adj.transpose()
        B_adj.conjugate()

        info("- Matrix B is assembled.")
        self._B = B
        self._B_adj = B_adj    

    def assemble_C(self):

        c = form(inner(self.u , self.v) * self.dx)
        C = assemble_matrix(c, self.bcs)
        C.assemble()
        info("- Matrix C is assembled.\n")
        self._C = C