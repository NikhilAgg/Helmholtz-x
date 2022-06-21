"""
A note on the nomenclature:
dim ~ dimensional quantity
ref ~ reference quantity for the non-dimensionalization
in ~ inlet, same as u ~ upstream (of the flame)
out ~ outlet, same as d ~ downstream (of the flame)
"""

from math import *
import numpy as np


r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

T_in = p_amb/(r*rho_u)  # [K]
T_out = p_amb/(r*rho_d)  # [K]

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

# print('Y_in =', Y_in)
# print('Y_out =', Y_out)

# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]
N = 0.014  # [/]

n = N*Q_tot/U_bulk  # [J/m]

# n_dim /= pi/4 * 0.047

"""[n_dim is case dependent]

n_dim = N*Q_tot/U_bulk  # [J/m]

1D - n_dim /= pi/4 * 0.047**2
2D - n_dim /= pi/4 * 0.047
3D - n_dim = n_dim
"""

tau = 0.0015  # [s]


x_f = np.array([[0.25, 0., 0.]])  # [m]
a_f = 0.025  # [m]

x_r = np.array([[0.20, 0., 0.]])  # [m]

# a_r = 0.0047  # [m]
a_r = 0.0047  # [m]

def gaussian(x,x_ref,sigma):
    first_term = 1/(sigma*np.sqrt(2*np.pi))
    second_term = np.exp(-1/2*((x[0]-x_ref)/(sigma))**2)
    return first_term*second_term

# def gaussian2D(x,x_ref,sigma,r):
#     first_term = 1/(sigma**2*np.sqrt(2*np.pi))
#     second_term = np.exp(((x[0]-x_ref)**2+r**2)//(-2*sigma**2))
#     return first_term*second_term

def gaussian2D(x, x_ref, sigma):
    first_term = 1/((2*np.pi)*sigma**2)
    second_term = np.exp(((x[0]-x_ref[0][0])**2+(x[1]-x_ref[0][1])**2)/(-2*sigma**2))
    return first_term*second_term

def w(mesh, x_r):
    V = FunctionSpace(mesh, ("CG", 1))
    w = Function(V)
    x = V.tabulate_dof_coordinates()
    w.interpolate(lambda x: gaussian(x,x_r,a_r))
    return w
    
def w2D(mesh, x_r):
    V = FunctionSpace(mesh, ("CG", 1))
    w = Function(V)
    x = V.tabulate_dof_coordinates()
    w.interpolate(lambda x: gaussian2D(x,x_r,a_r))
    return w

def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x[0]-x_f)/(sigma)))

def rho(mesh, x_f):

    V = FunctionSpace(mesh, ("CG", 1))
    rho = Function(V)
    x = V.tabulate_dof_coordinates()   

    rho.interpolate(lambda x: density(x, x_f, a_f, rho_d, rho_u))
    return rho

from dolfinx.fem import Function,FunctionSpace

def c(mesh, x_f):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[0]< x_f:
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
    rho_func = rho(mesh, x_f[0][0])
    w_func = w(mesh, x_r[0][0])
    w2D_func = w2D(mesh, x_r)
    c_func = c(mesh, x_f[0][0])
    
    with XDMFFile(MPI.COMM_WORLD, "Results/rho.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(rho_func)
        
    with XDMFFile(MPI.COMM_WORLD, "Results/w.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(w_func)
        
    with XDMFFile(MPI.COMM_WORLD, "Results/w2d.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(w2D_func)

    with XDMFFile(MPI.COMM_WORLD, "Results/c.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(c_func)
