import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from   sympy import sin, cos, latex
from   IPython.display import display, Math


def undo_abs(obj): 
    # remove abs value signs from trig functions: they cause problems with differentiation
    # this just handles one multiplier level
    if isinstance(obj, sy.Mul):
        val = 1
        for j, k in enumerate(obj.args):
            if isinstance(k, sy.Abs):
                val *= k.args[0]
            else:
                val *= k
        return val
    else:
        if isinstance(obj, sy.Abs): 
            val = obj.args[0]
            return val
    return obj


def pp(val):  # WS version of pretty-print
    # simplify each element of the object if applyfunc is supported by the object
    if 'applyfunc' in dir(val):
        val = val.applyfunc(sy.simplify)
    else:
        val = val.simplify()
    return display(Math(sy.latex(val)))


class Operators:
    
    def __init__(self, orig_coords, primed_coords, orig_from_primed):
        '''
        Generate the vector operators gradient, divergence, curl, Laplacian, in user-defined N-D coordinate transforms.
        
        Restrictions:
        - The original (unprimed) system must be Cartesian.
        - The primed system must have a diagonal metric (ie, the primed basis must be orthogonal).
        - The curl() operator is only calculated for 3D-to-3D coordinate transforms.
        
        Vector components are assumed to be 'normal': ie, the contravariant/covariant distinction is not required, 
        because normalization of the transformed vector bases is performed (eg, using the 'hi' terms of (1, 2), 
        which are the square roots of the diagonals of the metric). See Weinberg (3) for a discussion of these coordinates. 
        
        orig_coords      : list or tuple of sympy Symbol objects created with sympy symbols() function;
                           the original unprimed coordinates, must be Cartesian (ie, the metric is the identity matrix)
                           example: orig_coords = (x, y, z)
        primed_coords    : list or tuple of sympy Symbol objects created with sympy symbols() function;
                           the primed coordinates, with diagonal metric
                           example: primed_coords = (rho, phi, z) for 3D cylindrical coordinates
        orig_from_primed : list or tuple of dquations
                           the equations mapping the primed coords to the original coords;
                           example in 3D cylindrical coords: 
                           orig_from_primed = [rho*cos(phi), rho*sin(phi), z]
        NOTE: do not use underscores (representing subset characters) in the symbol-definition step: 
              symbols with underscores will clash with underscore symbol generation in _set_up_symbols() below.
              for example: 
              x0 = sympy.symbols('x0') is ok
              x0 = sympy.symbols('x_0') is NOT ok
        
        See https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates to validate vector operators
        
        Refs
        1) Arfken, George: 'Mathematical Methods for Physicists', second edition (1970), section 2.2
        2) Kreyszig xxx
        2) Weinberg xxx
        
        TODO:
        - fix problem in printing 3D Cartesian divergence and gradient(vector=False) case
        - improve printing for the vector operators
        - at present this assumes diagonal metrics for vector-operator calcs (as per Arfken/Weinberg developments)
          - at a future date, generalize to non-diagonal metrics
        '''
        self.coords    = orig_coords
        self.coords_p  = primed_coords
        self.transform = orig_from_primed
        self._set_up_symbols()
        self._create_operators()
        
    def _underscore_present(self, val):
        for k in val:
            if sy.latex(k).find('_') > 0:
                return True
        return False    

    def _set_up_symbols(self):
        
        if self._underscore_present(self.coords) or self._underscore_present(self.coords_p):
            raise ValueError(f"an underscore is present in orig_coords or primed_coords")
            
        # normalized Cartesian basis vectors
        dd = ['\\hat{\\mathbf{e_' + sy.latex(k) + '}}' for k in self.coords]
        self.Z = sy.Matrix(sy.symbols(dd))
        
        # normalized primed basis vectors
        pp = ['\\hat{\\mathbf{e_' + sy.latex(k) + '}}' for k in self.coords_p]
        self.Zp = sy.Matrix(sy.symbols(pp))

        # symbolic Laplacian operator
        self._Laplacian = sy.symbols('\\nabla^{2}', cls=sy.Function)
        
    def _scalar_field(self, F='F'):
        # create a generic scalard field with symbol 
        f = sy.symbols(F, cls=sy.Function, real=True)
        return f(*self.coords_p)
 
    def _vector_field(self, V='V'):
        # create a generic vector field with symbol V
        pp = [V + '_' + sy.latex(k) for k in self.coords_p]
        ss = []
        for k in pp:
            val = sy.symbols(k, cls=sy.Function, real=True)
            val = val(*self.coords_p)
            ss.append(val)
        return sy.Matrix(ss)

    def _create_operators(self):
        
        x                     = sy.Matrix(self.transform)
        y                     = sy.Matrix(self.coords_p)
        self._J               = x.jacobian(y)
        self._J_inv           = self._J.inv()
        self._metric          = sy.simplify(self._J.T * self._J)
        self._metric_sqrt     = self._metric.applyfunc(sy.sqrt).applyfunc(undo_abs)
        self._det             = self._metric.det()
        self._det_sqrt        = undo_abs(sy.sqrt(self._det))
        self._metric_inv      = self._metric.inv()
        self._metric_inv_sqrt = self._metric_sqrt.inv() # doing ops in this order works for diagonal metric only(?)

    @property
    def Jacobian(self):
        return self._J
    @property
    def Jacobian_inv(self):
        return self._J_inv
    @property
    def metric(self):
        return self._metric
    @property
    def metric_sqrt(self):
        return self._metric_sqrt
    @property
    def det(self):
        return self._det
    @property
    def det_sqrt(self):
        return self._det_sqrt
    @property
    def metric_inv(self):
        return self._metric_inv
    @property
    def metric_inv_sqrt(self):
        return self._metric_inv_sqrt
    @property
    def primed_basis(self):
        # transpose needed since tensor sum is over rows: see 6/12/22 handwritten;
        # need to normalize columns of Jacobian to get unit primed basis: using sqrt of metric inverse
        # (assuming diagonal metrics in this calculation)
        # i.e., the diagonals of the metric are the squared lengths of the unnormalized primed basis
        M = self._J * self.metric_inv_sqrt
        return M.T * self.Z

    def Laplacian_symbolic(self, coords):
        # symbolic representation of Laplacian
        if coords == 0: return 0
        return self._Laplacian(coords)
    
    def gradient(self, F=None, vector=True):
        # F is a scalar field in the primed coordinates; default is a symbolic scalar field
        # vector: True to return an n-element vector, False to return a symbolic represenation with unit vectors
        if F is None:
            F = self._scalar_field('f')
        V = sy.Matrix([F.diff(k) for k in self.coords_p])
        for j, k in enumerate(V):
            V[j] = k/self.metric_sqrt[j,j]
        V = sy.expand(V)
        if vector: return V
        return self.Zp.T * V

    def divergence(self, V=None):
        # V is a vector field in the primed coordinates; default is a symbolic vector field
        if V is None: V = self._vector_field('A')
        out = 0
        for j, k in enumerate(V):
            k   *= self.det_sqrt/self.metric_sqrt[j,j]
            out += k.diff(self.coords_p[j])/self.det_sqrt
        return sy.expand(out)

    def curl(self, V=None, vector=True):
        # V is a vector field in the primed coordinates; default is a symbolic vector field
        # vector: True to return an n-element vector, False to return a symbolic represenation with unit vectors
        if (len(self.coords) != 3) or (len(self.coords_p) != 3):
            raise ValueError(f"curl() requires a 3D-to-3D mapping")
        if V is None: V = self._vector_field('A')
        indx = {0:[1,2], 1:[2,0], 2:[0,1]}  # Levi-Civita positive entries
        h    = self.metric_sqrt
        out = []
        for i, m in indx.items():
            j, k = m
            p1 = (h[k,k] * V[k]).diff(self.coords_p[j])
            p2 = (h[j,j] * V[j]).diff(self.coords_p[k])
            out.append((h[i,i]/self.det_sqrt) * (p1 - p2))
        V = sy.expand(sy.Matrix(out))
        if vector: return V
        return self.Zp.T * V

    def material_derivative(self, A=None, B=None, vector=True):
        # see https://en.wikipedia.org/wiki/Material_derivative
        if A is None: A = self._vector_field('A')
        if B is None: B = self._vector_field('B')
        cp = self.coords_p
        nn = len(cp)
        h  = self.metric_sqrt
        out = []
        for j in range(nn):
            val = 0
            for i in range(nn):
                val += A[i] * B[j].diff(cp[i]) / h[i,i] +\
                       B[i] * (A[j] * h[j,j].diff(cp[i]) - A[i] * h[i,i].diff(cp[j])) / (h[i,i] * h[j,j])
            out.append(val)
        V = sy.expand(sy.Matrix(out))
        if vector: return V
        return self.Zp.T * V
 
    def Laplacian_scalar(self, F=None):
        # F is a scalar field in the primed coordinates; default is a symbolic scalar field
        if F is None: F = self._scalar_field('f')
        return sy.expand(sy.trigsimp(self.divergence(self.gradient(F, vector=True))))
    
    def Laplacian_vector_full(self, V=None): # use Arfken (1.80)
        # V is a vector field in the primed coordinates; default is a symbolic vector field
        if V is None: V = self._vector_field('A')
        return sy.expand(sy.trigsimp(self.gradient(self.divergence(V)) - self.curl(self.curl(V))))
    
    def Laplacian_vector_reduced(self, V=None):
        # V is a vector field in the primed coordinates; default is a symbolic vector field
        if V is None: V = self._vector_field('A')
        full = self.Laplacian_vector_full(V) # immutable matrix returned, can't modify
        out = []
        for j, k in enumerate(full):
            out.append(self.Laplacian_symbolic(V[j]) + k - self.Laplacian_scalar(V[j]))
        return sy.expand(sy.trigsimp(sy.Matrix(out)))

    # TODO: need to double-check this for correctness
    def vec_components_and_length(self, vec):
        vp =  self._J_inv * vec
        dd = [(j, k) for j, k in zip(self.coords, self.transform)]
        vp = vp.subs(dd) # make sure all variables are primed coordinates
        ll = vp.T * self._metric * vp
        orig_length = vec.T * vec  # assuming original system is Cartesian
        return vp, ll, orig_length
