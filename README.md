# vector-calculus-with-sympy
Implement vector-calculus operations with sympy. 

Generate the vector operators gradient, divergence, curl, Laplacian, etc.
from user-defined N-D coordinate transforms.

Restrictions:
- The original (unprimed) system must be Cartesian.
- The primed system must have a diagonal metric
  (ie, the primed basis must be orthogonal).
- The curl() operator is only calculated for 3D-to-3D coordinate transforms.

