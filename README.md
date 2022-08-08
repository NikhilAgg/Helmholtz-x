# Helmholtz-x
This python library implements the complex number version of Helmholtz Solver using dolfinx library.

It is using extensive parallelization with handled preallocations for generation of nonlinear part of the thermoacoustic Helmlholtz equation.

When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
