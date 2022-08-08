# Helmholtz-x
This python library implements the complex number version of Helmholtz Solver using dolfinx library.

It is using extensive parallelization with handled preallocations for generation of nonlinear part of the thermoacoustic Helmlholtz equation.

The nonlinear eigenproblem is solving using PETSc, SLEPc and FeniCS libraries. The discretized matrices are;

$$ A_{jk} = -\int_\Omega c^2\nabla \phi_j \cdot\nabla \phi_k dx   $$

$$ B_{jk} = \int_{\partial \Omega} \left( \frac{  ic}{Z}\right)  \phi_j\phi_k d{\sigma}   $$

$$ C_{jk} = \int_\Omega\phi_j\phi_k\ dx   $$



