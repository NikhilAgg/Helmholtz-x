# Helmholtz-x
This python library implements the complex number version of Helmholtz Solver using dolfinx library.

It is using extensive parallelization with handled preallocations for generation of nonlinear part of the thermoacoustic Helmlholtz equation.

When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 

$$ -\sum_{j=1}^{N}\int_\Omega c^2\nabla \phi_j \cdot\nabla \phi_k \dd{\vb{x}}\hat{p}_j +  \omega\sum_{j=1}^{N}\int_{\partial \Omega} \left( \frac{ic }{Z}\right)  \phi_j\phi_k \dd{\sigma}\hat{p}_j
+\omega^2\sum_{j=1}^{N} \int_\Omega\phi_j\phi_k\dd{\vb{x}}\hat{p}_j=0\\\textrm{(for k = 1, 2, 3, ..., N)  $$
