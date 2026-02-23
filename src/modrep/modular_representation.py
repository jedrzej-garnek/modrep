from sage.structure.sage_object import SageObject
from sage.modules.free_module import VectorSpace
from sage.matrix.special import identity_matrix

class ModularRepresentation(SageObject):
    """
    A class representing a modular representation of a group or algebra.
    
    The representation is defined by a set of matrices acting on a 
    finite-dimensional vector space over a finite field.
    """

    def __init__(self, generators, base_field=None):
        """
        Initialize the representation with a set of generating matrices.

        EXAMPLES:
            sage: F = GF(2)
            sage: m1 = matrix(F, [[0, 1], [1, 0]])
            sage: m2 = matrix(F, [[1, 1], [0, 1]])
            sage: rep = ModularRepresentation([m1, m2])
            sage: rep
            Modular representation of dimension 2 over Finite Field of size 2
            sage: rep.degree()
            2
        """
        if not generators:
            raise ValueError("At least one generator matrix is required.")
        
        # Ensure all generators are matrices over the same field
        first_gen = generators[0]
        if base_field is None:
            self._base_ring = first_gen.base_ring()
        else:
            self._base_ring = base_field
            
        self._degree = first_gen.nrows()
        
        # Cast generators to the correct space and ensure they are square
        self._generators = []
        for g in generators:
            if g.nrows() != g.ncols() or g.nrows() != self._degree:
                raise ValueError("Generators must be square matrices of the same dimension.")
            self._generators.append(g.change_ring(self._base_ring))
            
        # The underlying vector space container
        self._vector_space = VectorSpace(self._base_ring, self._degree)
        self._endomorphism_ring_basis = None

    def base_ring(self):
        """Return the base field of the representation."""
        return self._base_ring

    def degree(self):
        """Return the dimension of the underlying vector space."""
        return self._degree

    def generators(self):
        """Return the list of matrices generating the action."""
        return self._generators

    def vector_space(self):
        """Return the underlying Sage VectorSpace object."""
        return self._vector_space

    def _repr_(self):
        """Standard Sage string representation."""
        return f"Modular representation of dimension {self._degree} over {self._base_ring}"

    def endomorphism_ring_basis(self):
            """
            Compute a basis for the algebra of matrices commuting with all generators.
            
            This solves the system of linear equations X*G = G*X for all generators G.
            The result is a list of matrices that form a basis for the endomorphism ring.

            EXAMPLES:
                sage: F = GF(3)
                sage: g = matrix(F, [[1, 1], [0, 1]])
                sage: rep = ModularRepresentation([g])
                sage: basis = rep.endomorphism_ring_basis()
                sage: len(basis)
                2
                sage: all(b*g == g*b for b in basis)
                True
            """
            from sage.matrix.constructor import matrix
            from sage.modules.free_module import VectorSpace
            
            n = self.degree()
            K = self.base_ring()
            
            # We solve for the n^2 variables x_ij in the matrix X.
            # Each generator G gives n^2 equations from XG - GX = 0.
            
            equations = []
            for G in self.generators():
                # For each entry (r, c) in the matrix (XG - GX), we get a linear equation
                # (XG)_{rc} - (GX)_{rc} = sum_k (X_rk G_kc) - sum_k (G_rk X_kc) = 0
                for r in range(n):
                    for c in range(n):
                        eq = [K(0)] * (n * n)
                        # Contribution from XG: sum_k X_rk G_kc
                        for k in range(n):
                            eq[r * n + k] += G[k, c]
                        # Contribution from GX: sum_k G_rk X_kc
                        for k in range(n):
                            eq[k * n + c] -= G[r, k]
                        equations.append(eq)

            # Solve the system using Sage's kernel (nullspace) functionality
            system_matrix = matrix(K, equations)
            kernel = system_matrix.right_kernel()
            
            basis_matrices = []
            for vec in kernel.basis():
                # Reshape the (n^2) vector back into an (n x n) matrix
                basis_matrices.append(matrix(K, n, n, vec.list()))

            self._endomorphism_ring_basis = basis_matrices    
            return basis_matrices
    
    def endomorphism_algebra(self):
        """
        Return the EndomorphismAlgebra object for this representation.
        """
        basis = self.endomorphism_ring_basis()
        return EndomorphismAlgebra(basis)

    def is_indecomposable(self):
        """
        Check if the representation is indecomposable.
        
        Logic: Compute the endomorphism ring and check if it is a local ring.
        """
        return self.endomorphism_algebra().is_local()

from sage.matrix.constructor import matrix
from sage.modules.free_module import VectorSpace

class EndomorphismAlgebra(SageObject):
    """
    An associative algebra defined by a basis of matrices.
    """
    def __init__(self, basis):
        """
        Initialize the algebra from a basis of matrices.
        
        EXAMPLES:
            sage: F = GF(2)
            sage: b1 = matrix(F, [[1,0],[0,1]])
            sage: b2 = matrix(F, [[0,1],[0,0]])
            sage: alg = EndomorphismAlgebra([b1, b2])
            sage: alg.dimension()
            2
        """
        if not basis:
            raise ValueError("Basis cannot be empty.")
            
        self._basis = basis
        self._base_ring = basis[0].base_ring()
        self._dim = len(basis)
        self._matrix_dim = basis[0].nrows()
        
        # We pre-calculate a flat matrix of the basis to make 
        # "expressing a matrix as a linear combination" a simple back-solve.
        # Each column is a flattened basis matrix.
        self._basis_matrix = matrix(self._base_ring, 
                                    [b.list() for b in basis]).transpose()
        
        self._structure_constants = None

    def dimension(self):
        return self._dim

    def structure_constants(self):
        """
        Compute the structure constants (T) such that:
        basis[i] * basis[j] = sum_k T[i, j, k] * basis[k]
        
        Returns a dictionary or a dense array of coefficients.
        """
        if self._structure_constants is not None:
            return self._structure_constants

        d = self._dim
        # Initialize a 3D-like structure (list of matrices or dict)
        # T[i] is a matrix where the (j, k) entry is the coeff of basis[k] in B_i * B_j
        T = [] 
        
        for i in range(d):
            layer = []
            for j in range(d):
                product = self._basis[i] * self._basis[j]
                # Solve: self._basis_matrix * coefficients = product_vector
                coeffs = self._basis_matrix.solve_right(matrix(self._base_ring, product.list()).transpose())
                layer.append(coeffs.column(0).list())
            T.append(layer)
            
        self._structure_constants = T
        return T

    def _repr_(self):
        return f"Endomorphism algebra of dimension {self._dim} over {self._base_ring}"
    
    def jacobson_radical_basis(self):
        """
        Compute a basis for the Jacobson Radical J(A).
        
        This uses the trace form of the left-multiplication matrices.
        In a finite-dimensional algebra, an element x is in J(A) if 
        Tr(L_x * L_y) = 0 for all y in A, where L_x is the 
        left-multiplication map.

        EXAMPLES:
            sage: # Assuming 'alg' is the algebra from the previous step
            sage: rad_basis = alg.jacobson_radical_basis()
            sage: len(rad_basis) <= alg.dimension()
            True
        """
        d = self.dimension()
        K = self._base_ring
        T = self.structure_constants()
        
        # 1. Build the 'left multiplication' matrices L_i
        # L_i is the d x d matrix representing multiplication by basis[i]
        # (L_i)_{jk} is the coefficient of basis[k] in basis[i] * basis[j]
        left_mult_matrices = []
        for i in range(d):
            # T[i] is a d x d nested list where T[i][j] is the list of coeffs for basis[k]
            L_i = matrix(K, d, d, T[i])
            left_mult_matrices.append(L_i)
            
        # 2. Construct the Gram matrix G of the trace form
        # G_ij = Trace(L_i * L_j)
        G_rows = []
        for i in range(d):
            row = []
            for j in range(d):
                # Trace of the product of the two linear maps
                trace_val = (left_mult_matrices[i] * left_mult_matrices[j]).trace()
                row.append(trace_val)
            G_rows.append(row)
            
        G = matrix(K, G_rows)
        
        # 3. The radical is the kernel of this matrix
        kernel = G.right_kernel()
        
        # 4. Convert kernel vectors (coeffs) back into matrices in M_n(K)
        radical_basis = []
        for vec in kernel.basis():
            # Combine the original matrix basis using these coefficients
            # result = sum(vec[i] * basis[i])
            r_mat = sum(vec[i] * self._basis[i] for i in range(d))
            radical_basis.append(r_mat)
            
        return radical_basis
    
    def is_local(self):
        dim_A = self.dimension()
        dim_J = len(self.jacobson_radical_basis())
        # The quotient A/J is a division algebra. 
        # In modular reps over GF(q), if dim(A/J) == 1, it's definitely local.
        return (dim_A - dim_J) == 1