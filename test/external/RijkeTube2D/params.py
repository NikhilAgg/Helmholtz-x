from math import *
import numpy as np
from dolfinx.fem import Function,FunctionSpace
from helmholtz_x.parameters_utils import c_DG, rho,tau_linear,h,gaussianFunction,Q, Q_uniform, n_bump

# ------------------------------------------------------------

r_gas = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r_gas*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

c_u = sqrt(gamma*p_amb/rho_u)  # [kg/m^3]
c_d = sqrt(gamma*p_amb/rho_d)  # [kg/m^3]
# Reflection coefficients

T_u = c_u**2/(gamma*r_gas)
T_d = c_d**2/(gamma*r_gas)
# print(T_u,T_d)

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]


# ------------------------------------------------------------
# Flame transfer function

# Q_tot = 200.  # [W]
Q_tot = -27.008910380099735 # [W]

U_upstream = 0.10066660027273297
U_downstream = 0.14448618698798046 
U_bulk = 0.10066660027273297

FTF_mag =  0.1  # [/]
eta = FTF_mag * Q_tot / U_bulk
# print("ETA PRF", eta)

# LOTAN
# heat = 27.008910380099735
# mass = 2.1300000000000000E-004
# zmass = 0.38558287642205336#-1.82362764577375207E-002j
# zphi_phi = -zmass/mass
# constant = 1
# coeff = 195
# eta = heat / coeff * constant * zphi_phi 
# print("ETA LOTAN", eta)

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube / 4
eta /=  S_c

tau = 0.0015



x_f = np.array([0.25, 0., 0.])  # [m]
a_f = 0.025  # [m]

x_r = np.array([0.20, 0., 0.])  # [m]
a_r = 0.025

if __name__ == '__main__':

    from helmholtz_x.dolfinx_utils import XDMFReader,xdmf_writer
    RijkeTube2D = XDMFReader("MeshDir/rijke")
    mesh, subdomains, facet_tags = RijkeTube2D.getAll()

    rho_func = rho(mesh, x_f, a_f, rho_d, rho_u)
    w_func = gaussianFunction(mesh, x_r, a_r)
    h_func = gaussianFunction(mesh, x_f, a_f)
    
    c_func = c_DG(mesh, x_f, c_u, c_d)

    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)
