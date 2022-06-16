import dolfinx
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
#
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import FunctionSpace, Function, form
from ufl import dx,as_vector,inner,grad, TestFunction
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 200)
V = FunctionSpace(mesh, ("CG", 1))
alpha_r = 0.002
x_ref = 0.20

def gaussian(x,x_ref,sigma):
    first_term = 1/(sigma*np.sqrt(2*np.pi))
    second_term = np.exp(-1/2*((x[0]-x_ref)/(sigma))**2)
    return first_term*second_term


w = Function(V)
w.interpolate(lambda x: gaussian(x,x_ref,alpha_r))

meas = np.max(w.x.array)
temp = w.x.array
temp= temp/meas

w_normalized = Function(V) # Required for Parallel runs
w_normalized.vector.setArray(temp)
w_normalized.x.scatter_forward()


plt.plot(mesh.geometry.x, w_normalized.x.array.real)
plt.savefig("gaussian.pdf")
plt.clf()


def density(x, x_f, sigma, rho_d, rho_u):
    # rho_d + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x[0]-x_f)/(sigma)))

x_f = 0.25
rho_u = 1.38
rho_d = 0.975
alpha_f = 0.025
rho= Function(V)
rho.interpolate(lambda x: density(x, x_f, alpha_f, rho_d, rho_u))

plt.plot(mesh.geometry.x, rho.x.array.real)
plt.savefig("rho.pdf")
plt.clf()

n = as_vector([1])
v = TestFunction(V)

function = assemble_vector(form(inner(n,grad(v))*dx))
# function = assemble_vector(form(grad(v)*dx))

print(function.array)
# print(w_normalized.x.array)

right_vector = assemble_vector(form(inner(n,grad(v)*w_normalized/rho)*dx))
print(right_vector.array)


plt.plot(mesh.geometry.x, right_vector.array.real)
plt.savefig("right_vector.pdf")
plt.clf()