# Helmholtz-x
This python library implements the complex number version of Helmholtz Solver using dolfinx library.

It is using extensive parallelization with handled preallocations for generation of nonlinear part of the thermoacoustic Helmlholtz equation.

The nonlinear eigenproblem is solving using PETSc, SLEPc and FeniCS libraries. The thermoacoustic Helmholtz equation reads;

$$ A\textbf{P} + \omega B\textbf{P} + \omega^2C\textbf{P} = D(\omega)\textbf{P} $$

where 
$P$ is eigenvector, $\omega$ is eigenvalue ( $\frac{\omega}{2\pi}$ is eigenfrequency) and the discretized matrices are;

$$ A_{jk} = -\int_\Omega c^2\nabla \phi_j \cdot\nabla \phi_k dx   $$

$$ B_{jk} = \int_{\partial \Omega} \left( \frac{  ic}{Z}\right)  \phi_j\phi_k d{\sigma}   $$

$$ C_{jk} = \int_\Omega\phi_j\phi_k\ dx   $$

$$ D_{jk} = (\gamma-1) \frac{ q_{tot}  }{ U_{bulk}} \int_{\Omega} \phi_i n h(\textbf{x}) e^{i \omega \tau(\textbf{x})} d\textbf{x}  \int_{\Omega} \frac{w(\chi)}{\rho_0 (\chi)}  \nabla{\phi_j} \cdot \textbf{n}_{ref} d\chi $$

