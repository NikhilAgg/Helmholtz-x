import numpy as np

r_gas = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r_gas*rho_amb)  # [K]
c_amb = np.sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

c_u = np.sqrt(gamma*p_amb/rho_u)  
c_d = np.sqrt(gamma*p_amb/rho_d) 

T_u = c_u**2/(gamma*r_gas)
T_d = c_d**2/(gamma*r_gas)

# Reflection coefficients
R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# ------------------------------------------------------------
# Flame transfer function

FTF_mag =  0.1  # [/]
Q_tot = -27.008910380099735 # [W]
U_bulk = 0.10066660027273297

eta = FTF_mag * Q_tot / U_bulk
tau = 0.0015

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4
eta /=  S_c

x_f = np.array([0.25, 0., 0.])  # [m]
a_f = 0.025  # [m]

x_r = np.array([0.20, 0., 0.])  # [m]
a_r = 0.025

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from helmholtz_x.parameters_utils import c_step, rho, h, gaussianFunction
    from helmholtz_x.dolfinx_utils import OneDimensionalSetup

    n_elem = 3000
    mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

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
    
    c_plot = c_step(mesh, x_f, c_u, c_d)
    plt.plot(mesh.geometry.x, c_plot.x.array.real)
    plt.ylabel(r"$c$")
    plt.savefig("InputFunctions/c.png")
    plt.clf()
