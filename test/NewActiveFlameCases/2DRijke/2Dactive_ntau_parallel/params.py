"""
A note on the nomenclature:
dim ~ dimensional quantity
ref ~ reference quantity for the non-dimensionalization
in ~ inlet, same as u ~ upstream (of the flame)
out ~ outlet, same as d ~ downstream (of the flame)
"""

from math import *
import numpy as np
from dolfinx.fem import Function,FunctionSpace
from helmholtz_x.helmholtz_pkgx.helmholtz_utils import c_DG, rho,tau_linear,h,w,Q, Q_uniform, n_bump

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

# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]
n_value = 1

# For 2D dimensional consistency
d_tube = 0.047
n_2D = n_value #/((np.pi/4)*d_tube)

tau_u = 0.000
tau_d = 0.001

x_f = np.array([[0.25, 0., 0.]])  # [m]
a_f = 0.025  # [m]

x_r = np.array([[0.20, 0., 0.]])  # [m]
a_r = 0.0047  # [m]

a_h = a_f

if __name__ == '__main__':
    from mpi4py import MPI
    import dolfinx
    from helmholtz_x.geometry_pkgx.xdmf_utils import XDMFReader
    import ufl
    RijkeTube2D = XDMFReader("MeshDir/rijke")
    mesh, subdomains, facet_tags = RijkeTube2D.getAll()

    rho_func = rho(mesh, x_f, a_f, rho_d, rho_u)
    w_func = w(mesh, x_r, a_r)
    h_func = h(mesh, x_f, a_f)
    
    c_func = c_DG(mesh, x_f, c_in, c_out)

    from dolfinx.io import XDMFFile
    with XDMFFile(MPI.COMM_WORLD, "Results/rho.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(rho_func)
        
    with XDMFFile(MPI.COMM_WORLD, "Results/w.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(w_func)

    with XDMFFile(MPI.COMM_WORLD, "Results/h.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(h_func)  

    with XDMFFile(MPI.COMM_WORLD, "Results/c.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(c_func)
    
    tau_func = tau_linear(mesh,x_f, a_f, tau_u, tau_d)
    with XDMFFile(MPI.COMM_WORLD, "Results/tau.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(tau_func)

    q_func = Q(mesh, h_func, Q_tot)
    with XDMFFile(MPI.COMM_WORLD, "Results/q.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(q_func)
    integral_form = dolfinx.fem.form(q_func*ufl.dx)
    print("Q gaussian: ", dolfinx.fem.assemble_scalar(integral_form))

    q_uniform = Q_uniform(mesh, subdomains, Q_tot)
    q_uniform_func = n_bump(mesh,x_f,a_f,q_uniform.value)
    with XDMFFile(MPI.COMM_WORLD, "Results/q_uniform.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(q_uniform_func)
    q_uniform_integral_form = dolfinx.fem.form(q_uniform_func*ufl.dx)
    print("Q gaussian: ", dolfinx.fem.assemble_scalar(integral_form))