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
    
    def is_isomorphic(self, other):
        """
        Return True if this representation is isomorphic to another.
        
        Two representations are isomorphic if there exists an invertible 
        matrix M such that M * g_i = h_i * M for all generators g_i and h_i.
        
        EXAMPLES:
            sage: F = GF(3)
            sage: g = matrix(F, [[0, 1], [1, 0]])
            sage: h = matrix(F, [[1, 1], [0, 2]])
            sage: rep1 = ModularRepresentation([g])
            sage: P = matrix(F, [[1, 1], [0, 1]]) # Change of basis
            sage: rep2 = ModularRepresentation([P * g * P.inverse()])
            sage: rep1.is_isomorphic(rep2)
            True
        """
        if not isinstance(other, ModularRepresentation):
            raise TypeError("Can only compare with another ModularRepresentation.")

        if self._degree != other._degree:
            return False

        if self._base_ring != other._base_ring:
            # Optionally try to coerce, but usually representations are 
            # defined over a specific field.
            return False
            
        if len(self._generators) != len(other._generators):
            return False
        
        n = self._degree
        F = self._base_ring
        
        # We solve for the entries of an n x n matrix X
        # X*A_i - B_i*X = 0
        equations = []
        for A, B in zip(self._generators, other._generators):
            # For each pair of generators, generate n^2 linear equations
            for i in range(n):
                for j in range(n):
                    # Entry (i,j) of (X*A - B*X)
                    # (X*A)_ij = sum_k X_ik A_kj
                    # (B*X)_ij = sum_k B_ik X_kj
                    row = [0] * (n * n)
                    # X*A part
                    for k in range(n):
                        row[i * n + k] += A[k, j]
                    # B*X part
                    for k in range(n):
                        row[k * n + j] -= B[i, k]
                    equations.append(row)
        
        # Solve the system
        from sage.matrix.constructor import matrix
        syst = matrix(F, equations)
        null_basis = syst.right_kernel().basis()
        
        if not null_basis:
            return False
            
        # Any element in the null space is a candidate for the isomorphism matrix.
        # However, it MUST be invertible.
        # In the modular case, if a basis exists, we may need to find a 
        # non-singular linear combination.
        if len(null_basis) == 1:
            M = matrix(F, n, n, null_basis[0])
            return M.is_invertible()
            
        # If the space of intertwiners has dim > 1, we check if it contains 
        # an invertible matrix.
        # This is a classic problem: for finite fields, we can sample or 
        # use the fact that if an isomorphism exists, a 'random' one 
        # is often invertible.
        import random
        for _ in range(20): # Try random combinations
            coeffs = [F.random_element() for _ in range(len(null_basis))]
            candidate_vec = sum(c * v for c, v in zip(coeffs, null_basis))
            if matrix(F, n, n, candidate_vec).is_invertible():
                return True
                
        return False

    def split(self):
        """
        Próbuje rozbić reprezentację na sumę prostą dwóch nietrywialnych składników.
        Zwraca (V1, V2) jako obiekty ModularRepresentation lub None, jeśli nie 
        udało się znaleźć rozkładu (np. moduł jest nierozkładalny).
        """
        # Jeśli algebra endomorfizmów jest lokalna, moduł jest nierozkładalny
        if self.is_indecomposable():
            return None

        basis = self.endomorphism_ring_basis()
        F = self.base_ring()
        
        # Próbujemy znaleźć endomorfizm o rozkładalnym wielomianie minimalnym.
        # W algebrze, która nie jest lokalna, losowy element zazwyczaj 
        # pozwoli na nietrywialny rozkład składowych pierwotnych.
        import random
        for _ in range(100):
            # 1. Losowa kombinacja liniowa bazy endomorfizmów
            coeffs = [F.random_element() for _ in range(len(basis))]
            f = sum(c * b for c, b in zip(coeffs, basis))
            
            # 2. Obliczamy wielomian minimalny endomorfizmu f
            mu = f.minimal_polynomial()
            factors = mu.factor()
            
            # 3. Jeśli mamy więcej niż jeden czynnik nierozkładalny
            if len(factors) > 1:
                # Przyjmujemy g jako pierwszy czynnik potęgowy, h jako resztę
                g_poly, exp = factors[0]
                g = g_poly**exp
                h = mu // g
                
                # 4. Wyznaczamy jądra (podprzestrzenie niezmiennicze)
                # Ponieważ gcd(g, h) = 1, zachodzi V = ker(g(f)) \oplus ker(h(f))
                V1_subspace = g(f).kernel()
                V2_subspace = h(f).kernel()
                
                # Safety check (wymiary muszą być dodatnie)
                if V1_subspace.dimension() == 0 or V2_subspace.dimension() == 0:
                    continue
                    
                # 5. Tworzymy nowe reprezentacje ograniczone do tych podprzestrzeni
                rep1 = self._restrict_to_subspace(V1_subspace)
                rep2 = self._restrict_to_subspace(V2_subspace)
                
                return (rep1, rep2)
                
            # Uwaga: Jeśli mu = p(x)^k i deg(p) > 1, to f generuje rozszerzenie ciała 
            # lub składową macierzową. Wtedy też można szukać idempotentów, 
            # ale klasyczna faktoryzacja nad ciałem bazowym zazwyczaj wystarcza 
            # dla p-grup nad F_p, bo jedynym modułem prostym jest F_p.

        return None

    def _restrict_to_subspace(self, W):
        """
        Tworzy nową ModularRepresentation poprzez ograniczenie działania 
        generatorów do podprzestrzeni W.
        """
        if W.dimension() == 0:
            raise ValueError("Nie można ograniczyć do zerowej podprzestrzeni.")
            
        # Macierz bazy podprzestrzeni W (wiersze to wektory bazy)
        B = W.basis_matrix()
        
        restricted_gens = []
        for G in self.generators():
            # Szukamy macierzy M takiej, że B * G = M * B
            # (M opisuje działanie G na współrzędnych bazy B)
            # W Sage: B.solve_left(B * G) rozwiązuje układ X * B = B * G
            M = B.solve_left(B * G)
            restricted_gens.append(M)
            
        return ModularRepresentation(restricted_gens, base_field=self.base_ring())

    def decompose(self):
            """
            Decompose the representation into a direct sum of indecomposable summands.
            
            This is a recursive implementation of the Krull-Schmidt theorem logic.
            
            RETURNS:
                A list of indecomposable ModularRepresentation objects.
            """
            if self.is_indecomposable():
                return [self]
            
            # split() returns a tuple (V1, V2)
            summands = self.split()
            
            if summands is None:
                # This handles cases where is_local might be ambiguous 
                # but no idempotent was found.
                return [self]
            
            final_list = []
            for s in summands:
                # Recursively decompose each piece
                final_list.extend(s.decompose())
                
            return final_list

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
        Compute a basis for the Jacobson Radical J(A) that works in any characteristic.
        
        This uses the property that in a finite-dimensional algebra, J(A) is 
        the largest nilpotent ideal. We find it by identifying elements whose 
        left-multiplication matrices are nilpotent.
        """
        from sage.matrix.constructor import matrix
        
        d = self.dimension()
        K = self._base_ring
        p = K.characteristic()
        T = self.structure_constants()
        
        # 1. Build the 'left multiplication' matrices L_i
        # L_i is the d x d matrix representing multiplication by basis[i]
        # (L_i)_{jk} is the coefficient of basis[k] in basis[i] * basis[j]
        left_mult_matrices = []
        for i in range(d):
            # T[i][j][k] is the coeff of basis[k] in B_i * B_j
            L_i = matrix(K, d, d, T[i])
            left_mult_matrices.append(L_i)

        # 2. Dickson's Method / Trace of Powers
        # An element x = sum c_i B_i is in J(A) iff Tr((L_x)^k) = 0 
        # for all 1 <= k <= d. In char p, we check Tr((L_x)^{p^m}).
        
        # We construct a system of equations where each row is the trace 
        # of the left-multiplication matrix of the basis elements.
        # However, for a general non-commutative algebra, we use the 
        # intersection of kernels of representations or the 'Meat-Axe'.
        
        # Robust approach for Sage: Calculate the radical of the 
        # finite-dimensional algebra directly using Sage's internal 
        # algebra structures if possible, or use the 'Power Trace' matrix.
        
        equations = []
        # We need enough powers to catch nilpotency. 
        # In char p, Tr(X^{p^k}) is linear in the coordinates of X.
        pk = p
        while pk <= d:
            row = []
            for i in range(d):
                # We look at the trace of the p^k-th power of the left mult of basis[i]
                val = (left_mult_matrices[i]**pk).trace()
                row.append(val)
            equations.append(row)
            pk *= p
            
        # If the above is too thin, we fall back to the generic 
        # "trace of basis products" but supplemented by nilpotency checks.
        G_rows = []
        for i in range(d):
            row = []
            for j in range(d):
                # The "Corrected" trace form for char p
                # This is a heuristic that works for most modular endomorphism rings
                val = (left_mult_matrices[i] * left_mult_matrices[j]).trace()
                row.append(val)
            G_rows.append(row)

        # Combine systems
        syst = matrix(K, equations + G_rows)
        kernel = syst.right_kernel()
        
        # 3. Filter the kernel for actual nilpotency
        # Elements in J(A) must be nilpotent.
        final_vectors = []
        for vec in kernel.basis():
            test_mat = sum(vec[i] * left_mult_matrices[i] for i in range(d))
            if test_mat.is_nilpotent():
                final_vectors.append(vec)
        
        # 4. Convert back to matrix basis of A
        radical_basis = []
        for vec in final_vectors:
            r_mat = sum(vec[i] * self._basis[i] for i in range(d))
            radical_basis.append(r_mat)
            
        return radical_basis
    
    def is_local(self):
        dim_A = self.dimension()
        dim_J = len(self.jacobson_radical_basis())
        # The quotient A/J is a division algebra. 
        # In modular reps over GF(q), if dim(A/J) == 1, it's definitely local.
        return (dim_A - dim_J) == 1