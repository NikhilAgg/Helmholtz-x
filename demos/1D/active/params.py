"""
A note on the nomenclature:
dim ~ dimensional quantity
ref ~ reference quantity for the non-dimensionalization
in ~ inlet, same as u ~ upstream (of the flame)
out ~ outlet, same as d ~ downstream (of the flame)
"""

from math import *
import numpy as np

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

# Specific impedance

Z_in = (1 + R_in)/(1 - R_in)
Z_out = (1 + R_out)/(1 - R_out)

# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]
N = 0.014  # [/]

n_dim = N*Q_tot/U_bulk  # [J/m]

n_dim /= pi/4 * 0.047**2

"""[n_dim is case dependent]

n_dim = N*Q_tot/U_bulk  # [J/m]

1D - n_dim /= pi/4 * 0.047**2
2D - n_dim /= pi/4 * 0.047
3D - n_dim = n_dim
"""

# print('n_dim = %f' % n_dim)

tau_dim = 0.0015  # [s]

# ------------------------------------------------------------

x_f_dim = np.array([[0.25, 0., 0.]])  # [m]
a_f_dim = 0.025  # [m]

x_r_dim = np.array([[0.20, 0., 0.]])  # [m]

# Non-dimensionalization

U_ref = c_amb  # [m/s]
p_ref = p_amb  # [Pa]

# ------------------------------------------------------------

rho_in = rho_in_dim*U_ref**2/p_ref
rho_out = rho_out_dim*U_ref**2/p_ref

c_in = c_in_dim/U_ref
c_out = c_out_dim/U_ref

# ------------------------------------------------------------

n = n_dim/(p_ref*L_ref**2)

tau = tau_dim*U_ref/L_ref

# ------------------------------------------------------------

x_f = x_f_dim/L_ref
x_r = x_r_dim/L_ref

a_f = a_f_dim/L_ref

# ------------------------------------------------------------

from dolfinx.fem import Function,FunctionSpace

def c(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    global c_in
    global c_out
    global x_f
    global a_f
    x_f = x_f[0][0]
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[0]< x_f:
            c.vector.setValueLocal(i, c_in)
        else:
            c.vector.setValueLocal(i, c_out)
    return c
