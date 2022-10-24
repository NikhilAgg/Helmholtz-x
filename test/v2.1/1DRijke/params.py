from math import *
import numpy as np
from dolfinx.fem import Function,FunctionSpace
from helmholtz_x.parameters_utils import c_DG, rho,tau_linear,h,gaussianFunction,Q, Q_uniform, n_bump

# ------------------------------------------------------------

L_ref = 1.  # [m]

# ------------------------------------------------------------

r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_in_dim = rho_amb  # [kg/m^3]
rho_out_dim = 0.85  # [kg/m^3]

c_in_dim = sqrt(gamma*p_amb/rho_in_dim)  # [kg/m^3]
c_out_dim = sqrt(gamma*p_amb/rho_out_dim)  # [kg/m^3]

# Reflection coefficients

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]


# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]


FTF_mag =  0.014  # [/]
eta = FTF_mag * Q_tot / U_bulk

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4

eta /=  S_c

tau_dim = 0.0015



x_f_dim = np.array([0.25, 0., 0.])  # [m]
a_f_dim = 0.025  # [m]

x_r_dim = np.array([0.20, 0., 0.])  # [m]
a_r_dim = 0.025

# Non-dimensionalization

U_ref = c_amb  # [m/s]
p_ref = p_amb  # [Pa]

rho_u = rho_in_dim*U_ref**2/p_ref
rho_d = rho_out_dim*U_ref**2/p_ref

c_u = c_in_dim/U_ref
c_d = c_out_dim/U_ref

# ------------------------------------------------------------

eta = eta/(p_ref*L_ref**2)

tau = tau_dim*U_ref/L_ref

# ------------------------------------------------------------

x_f = x_f_dim/L_ref
x_r = x_r_dim/L_ref

a_f = a_f_dim/L_ref
a_r = a_r_dim/L_ref


if __name__ == '__main__':
    from mpi4py import MPI
    import dolfinx
    import matplotlib.pyplot as plt
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 200)
    V = FunctionSpace(mesh, ("CG", 1))

    w_plot = gaussianFunction(mesh, x_r, a_r)
    plt.plot(mesh.geometry.x, w_plot.x.array.real)
    plt.ylabel("w")
    plt.savefig("Results/w.png")
    plt.clf()

    h_plot = h(mesh, x_f, a_f)
    plt.plot(mesh.geometry.x, h_plot.x.array.real)
    plt.ylabel("h")
    plt.savefig("Results/h.png")
    plt.clf()

    rho_plot = rho(mesh, x_f, a_f, rho_d, rho_u)
    plt.plot(mesh.geometry.x, rho_plot.x.array.real)
    plt.ylabel(r"$\rho$")
    plt.savefig("Results/rho.png")
    plt.clf()
    
