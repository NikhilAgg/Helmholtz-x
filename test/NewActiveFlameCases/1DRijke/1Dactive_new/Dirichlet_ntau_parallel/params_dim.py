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

# c_in = sqrt(gamma*p_amb/rho_u)  # [kg/m^3]
# c_out = sqrt(gamma*p_amb/rho_d)  # [kg/m^3]
c_in = 400
c_out = 400
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
# N = 0.014  # [/]
n_value = 1

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4
n_1D = n_value # / S_c

tau_u = 0.0
tau_d = 0.001

shifter = 0.0

x_f = np.array([[0.25+shifter, 0., 0.]])  # [m]
a_f = 0.025  # [m]

x_r = np.array([[0.20+shifter, 0., 0.]])  # [m]
a_r = 0.0047


a_v = a_f
# ------------------------------------------------------------

def gaussian(x,x_ref,sigma):
    first_term = 1/(sigma*np.sqrt(2*np.pi))
    second_term = np.exp(-1/2*((x[0]-x_ref)/(sigma))**2)
    return first_term*second_term

def w(mesh, x_r):
    V = FunctionSpace(mesh, ("CG", 1))
    w = Function(V)
    x = V.tabulate_dof_coordinates()
    w.interpolate(lambda x: gaussian(x,x_r,a_r))
    return w

def h(mesh, x_f):
    V = FunctionSpace(mesh, ("CG", 1))
    v = Function(V)
    x = V.tabulate_dof_coordinates()
    v.interpolate(lambda x: gaussian(x,x_f,a_v))
    return v
    
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x[0]-x_f)/(sigma)))

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
        if midpoint[0] > x_f-a_f and midpoint[0]< x_f+a_f :
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
        
        if midpoint[0] < x_f-a_f:
            tau.vector.setValueLocal(i, tau_u)
        elif midpoint[0] >= x_f-a_f and midpoint[0]<= x_f+a_f :
            tau.vector.setValueLocal(i, tau_u+(tau_d-tau_u)/(2*a_f)*(midpoint[0]-(x_f-a_f)))
        else:
            tau.vector.setValueLocal(i, tau_d)
    return tau

def c(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    # global c_in
    # global c_out
    # global x_f
    x_f = x_f[0][0]
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
    import matplotlib.pyplot as plt
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 200)
    V = FunctionSpace(mesh, ("CG", 1))

    w_plot = w(mesh,x_r[0][0])
    plt.plot(mesh.geometry.x, w_plot.x.array.real)
    plt.ylabel("w")
    plt.savefig("w.png")
    plt.clf()

    h_plot = h(mesh,x_f[0][0])
    plt.plot(mesh.geometry.x, h_plot.x.array.real)
    plt.ylabel("h")
    plt.savefig("h.png")
    plt.clf()

    rho_plot = rho(mesh,x_f[0][0])
    plt.plot(mesh.geometry.x, rho_plot.x.array.real)
    plt.ylabel(r"$\rho$")
    plt.savefig("rho.png")
    plt.clf()
    
    n_plot = n(mesh,x_f[0][0])
    plt.plot(mesh.geometry.x, n_plot.x.array.real)
    plt.ylabel(r"$n$")
    plt.savefig("n.png")
    plt.clf()

    tau_plot = tau(mesh,x_f[0][0])
    plt.plot(mesh.geometry.x, tau_plot.x.array.real)
    plt.ylabel(r"$\tau$")
    plt.savefig("tau.png")
