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
print(T_u,T_d)

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]


# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]
# N = 0.014  # [/]


FTF_mag =  0.014  # [/]
eta = FTF_mag * Q_tot / U_bulk

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4

eta /=  S_c

tau = 0.0015



x_f = np.array([0.25, 0., 0.])  # [m]
a_f = 0.025  # [m]

x_r = np.array([0.20, 0., 0.])  # [m]
a_r = 0.025

if __name__ == '__main__':
    from mpi4py import MPI
    import dolfinx
    import matplotlib.pyplot as plt
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 200)
    V = FunctionSpace(mesh, ("CG", 1))

    w_plot = gaussianFunction(mesh, x_r, a_r)
    plt.plot(mesh.geometry.x, w_plot.x.array.real)
    plt.ylabel("w")
    plt.savefig("InputFunctions/w.png")
    plt.clf()

    h_plot = h(mesh, x_f, a_f)
    plt.plot(mesh.geometry.x, h_plot.x.array.real)
    plt.ylabel("h")
    plt.savefig("InputFunctions/h.png")
    plt.clf()

    rho_plot = rho(mesh, x_f, a_f, rho_d, rho_u)
    plt.plot(mesh.geometry.x, rho_plot.x.array.real)
    plt.ylabel(r"$\rho$")
    plt.savefig("InputFunctions/rho.png")
    plt.clf()
    
    c_plot = c_DG(mesh, x_f, c_u, c_d)
    plt.plot(mesh.geometry.x, c_plot.x.array.real)
    plt.ylabel(r"$c$")
    plt.savefig("InputFunctions/c.png")
    plt.clf()

    # tau_plot = tau_linear(mesh,x_f, a_f, tau_u, tau_d)
    # plt.plot(mesh.geometry.x, tau_plot.x.array.real)
    # plt.ylabel(r"$\tau$")
    # plt.savefig("InputFunctions/tau.png")
