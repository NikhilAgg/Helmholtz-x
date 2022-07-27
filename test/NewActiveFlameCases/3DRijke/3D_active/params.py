from math import *
import numpy as np
from dolfinx.fem import Function,FunctionSpace
# ------------------------------------------------------------

r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

c_in = sqrt(gamma*p_amb/rho_u)  # [kg/m^3]
c_out = sqrt(gamma*p_amb/rho_d)  # [kg/m^3]

# Reflection coefficients

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# Specific impedance

Z_in = (1 + R_in)/(1 - R_in)
Z_out = (1 + R_out)/(1 - R_out)


# Specific admittance

Y_in = 1/Z_in
Y_out = 1/Z_out

# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]

n_value = 1  # [J/m]

tau_u = 0.000
tau_d = 0.001  # [s]


x_f = np.array([[0., 0., 0.25]])  # [m]
a_f = 0.025  # [m]

x_r = np.array([[0., 0., 0.20]])  # [m]
a_r = 0.0047
# ------------------------------------------------------------

def gaussian(x,x_ref,sigma):
    first_term = 1/(sigma*np.sqrt(2*np.pi))
    second_term = np.exp(-1/2*((x[2]-x_ref)/(sigma))**2)
    return first_term*second_term

def w(mesh, x_r):
    V = FunctionSpace(mesh, ("CG", 1))
    w = Function(V)
    x = V.tabulate_dof_coordinates()

    # global x_r
    # global a_r
    w.interpolate(lambda x: gaussian(x,x_r,a_r))
    return w
    
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x[2]-x_f)/(sigma)))

def rho(mesh, x_f):

    V = FunctionSpace(mesh, ("CG", 1))
    rho = Function(V)
    x = V.tabulate_dof_coordinates()   

    rho.interpolate(lambda x: density(x, x_f, a_f, rho_d, rho_u))
    return rho


def n(mesh, x_f):
    V = FunctionSpace(mesh, ("CG", 1))
    n = Function(V)
    x = V.tabulate_dof_coordinates()   

    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[2] >= x_f-a_f and midpoint[2]<= x_f+a_f :
            n.vector.setValueLocal(i, n_value)
        else:
            n.vector.setValueLocal(i, 0.)
    return n

def tau(mesh, x_f):
    V = FunctionSpace(mesh, ("CG", 1))
    tau = Function(V)
    x = V.tabulate_dof_coordinates()   
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        
        if midpoint[2] < x_f-a_f:
            tau.vector.setValueLocal(i, tau_u)
        elif midpoint[2] >= x_f-a_f and midpoint[2]<= x_f+a_f :
            tau.vector.setValueLocal(i, tau_u+(tau_d-tau_u)/(2*a_f)*(midpoint[2]-(x_f-a_f)))
        else:
            tau.vector.setValueLocal(i, tau_d)
    return tau

from dolfinx.fem import Function,FunctionSpace

def c(mesh, x_f):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[2]< x_f:
            c.vector.setValueLocal(i, c_in)
        else:
            c.vector.setValueLocal(i, c_out)
    return c

if __name__ == '__main__':
    from mpi4py import MPI
    import dolfinx
    from dolfinx.io import XDMFFile
    from helmholtz_x.geometry_pkgx.xdmf_utils import XDMFReader

    RijkeTube3D = XDMFReader("MeshDir/rijke")
    mesh, subdomains, facet_tags = RijkeTube3D.getAll()
    print(x_f,x_r)
    rho_func = rho(mesh, x_f[0][2])
    w_func = w(mesh, x_r[0][2])
    c_func = c(mesh, x_f[0][2])
    n_func = n(mesh, x_f[0][2])
    tau_func = tau(mesh, x_f[0][2])
    
    with XDMFFile(MPI.COMM_WORLD, "Results/rho.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(rho_func)
        
    with XDMFFile(MPI.COMM_WORLD, "Results/w.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(w_func)

    with XDMFFile(MPI.COMM_WORLD, "Results/c.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(c_func)
        
    with XDMFFile(MPI.COMM_WORLD, "Results/n.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(n_func)
    
    with XDMFFile(MPI.COMM_WORLD, "Results/tau.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(tau_func)
