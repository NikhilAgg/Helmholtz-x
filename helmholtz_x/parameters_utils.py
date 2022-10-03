from dolfinx.fem import FunctionSpace, Function, form, Constant, assemble_scalar
from .dolfinx_utils import normalize
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from ufl import dx, Measure

def gaussian(x,x_ref,sigma):
    first_term = 1/(sigma*np.sqrt(2*np.pi))
    second_term = np.exp(-1/2*((x[0]-x_ref)/(sigma))**2)
    return first_term*second_term

def gaussian2D(x, x_ref, sigma):
    first_term = 1/((2*np.pi)*sigma**2)
    second_term = np.exp(((x[0]-x_ref[0][0])**2+(x[1]-x_ref[0][1])**2)/(-2*sigma**2))
    return first_term*second_term

def gaussian3D(x,x_ref,sigma):
    # https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    N = 1/(sigma**3*(2*np.pi)**(3/2))
    spatials = (x[0]-x_ref[0])**2 + (x[1]-x_ref[1])**2 + (x[2]-x_ref[2])**2
    exp = np.exp(-spatials/(2*sigma**2))
    return N*exp

def gaussianCroci(x,x_ref,tightness):
    x_term = -tightness*(x[0]-x_ref[0][0])**2
    y_term = -tightness*(abs(x[1])-x_ref[0][1])**2 # abs() can be add
    return np.exp(x_term + y_term)

def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))

def rho(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    rho = Function(V)
    # x = V.tabulate_dof_coordinates()   
    if mesh.geometry.dim == 1 or mesh.geometry.dim == 2:
        x_f = x_f[0][0]
        rho.interpolate(lambda x: density(x[0], x_f, a_f, rho_d, rho_u))
    elif mesh.geometry.dim == 3:
        x_f = x_f[0][2]
        rho.interpolate(lambda x: density(x[2], x_f, a_f, rho_d, rho_u))
    return rho

def rho_ideal(mesh, temperature, P_amb, R):
    V = FunctionSpace(mesh, ("DG", 0))
    density = Function(V)
    density.x.array[:] =  P_amb /(R * temperature.x.array)
    density.x.scatter_forward()
    return density

def gaussianFunction(mesh, x_r, a_r, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    # x = V.tabulate_dof_coordinates()
    if mesh.geometry.dim == 1:
        x_r = x_r[0][0]
        w.interpolate(lambda x: gaussian(x,x_r,a_r))
    elif mesh.geometry.dim == 2:
        # w.interpolate(lambda x: gaussian2D(x,x_r,a_r))
        x_r = x_r[0][0]
        w.interpolate(lambda x: gaussian(x,x_r,a_r))
    # elif mesh.geometry.dim == 3:
    #     x_r = x_r[0][2]
    #     w.interpolate(lambda x: gaussian3D(x,x_r,a_r))
    elif mesh.geometry.dim == 3:
        w.interpolate(lambda x: gaussian3D(x,x_r,a_r))
    w = normalize(w)
    return w

def w_croci(mesh, x_ref, tightness, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)

    if mesh.geometry.dim == 2 :
        w.interpolate(lambda x: gaussianCroci(x,x_ref,tightness))
    else:
        ValueError("Only 2D cases can be implemented.")
    
    w_normalized = normalize(w)
    return w_normalized

def h(mesh, x_f, a_f, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    h = Function(V)
    # x = V.tabulate_dof_coordinates()
    if mesh.geometry.dim == 1:
        x_f = x_f[0][0]
        h.interpolate(lambda x: gaussian(x,x_f,a_f))
    elif mesh.geometry.dim == 2:
        # h.interpolate(lambda x: gaussian2D(x,x_f,a_f))
        x_f = x_f[0][0]
        h.interpolate(lambda x: gaussian(x,x_f,a_f))
    elif mesh.geometry.dim == 3:
        x_f = x_f[0][2]
        h.interpolate(lambda x: gaussian3D(x,x_f,a_f))
    return h

def tau_linear(mesh, x_f, a_f, tau_u, tau_d, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    tau = Function(V)
    x = V.tabulate_dof_coordinates()  
    if mesh.geometry.dim == 1:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 2:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 3:
        x_f = x_f[0][2] 
        axis = 2
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        
        if midpoint[axis] < x_f-a_f:
            tau.vector.setValueLocal(i, tau_u)
        elif midpoint[axis] >= x_f-a_f and midpoint[axis]<= x_f+a_f :
            tau.vector.setValueLocal(i, tau_u+(tau_d-tau_u)/(2*a_f)*(midpoint[axis]-(x_f-a_f)))
        else:
            tau.vector.setValueLocal(i, tau_d)
    return tau

def n_bump(mesh, x_f, a_f, n_value, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    n = Function(V)
    x = V.tabulate_dof_coordinates()   
    if mesh.geometry.dim == 1:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 2:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 3:
        x_f = x_f[0][2]
        axis = 2
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[axis] > x_f-a_f and midpoint[axis]< x_f+a_f :
            n.vector.setValueLocal(i, n_value)
        else:
            n.vector.setValueLocal(i, 0.)
    return n

def c_DG(mesh, x_f, c_u, c_d):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    if mesh.geometry.dim == 1:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 2:
        x_f = x_f[0][0]
        axis = 0
    elif mesh.geometry.dim == 3:
        x_f = x_f[0][2]
        axis = 2
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[axis]< x_f:
            c.vector.setValueLocal(i, c_u)
        else:
            c.vector.setValueLocal(i, c_d)
    return c

def sound_speed(mesh, temperature):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    if isinstance(temperature, Function):
        c.x.array[:] =  20.05 * np.sqrt(temperature.x.array)
    else:
        c.x.array[:] =  20.05 * np.sqrt(temperature)
    c.x.scatter_forward()
    return c

def Q(mesh, h, Q_total, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    q = Function(V)
    volume_form = form(Constant(mesh, PETSc.ScalarType(1))*dx)
    V_flame = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
    q_tot = Q_total/V_flame

    q.x.array[:] = q_tot*h.x.array[:]
    q.x.scatter_forward()

    return q

def Q_uniform(mesh, subdomains, Q_total, flame_tag=0, degree=1):

    dx = Measure("dx", subdomain_data=subdomains)
    volume_form = form(Constant(mesh, PETSc.ScalarType(1))*dx(flame_tag))
    V_flame = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
    q_uniform = Constant(mesh, PETSc.ScalarType(Q_total/V_flame))

    return q_uniform