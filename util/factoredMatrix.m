classdef factoredMatrix
    %FACTOREDMATRIX An n x n matrix M = Z*Zᵀ stored in terms of its square-root factors.
    %   Matrix Lyapunov and Riccati equations are ubiquitous in control and
    %   model reduction. Often though, it can be beneficial numerically to deal
    %   with their square-root factors. For example in balanced truncation, the
    %   balancing transformation can be constructed directly from the
    %   square-root factors [1], and additional accuracy can be retained by
    %   using lyapchol() [2] rather than lyap() followed by chol() [3], even
    %   permitting the reduction of non-minimal systems [1]. In control theory,
    %   one of the way that modern solvers (such as those in the M-M.E.S.S.
    %   package [4]) can scale much higher than traditional solvers such as
    %   Matlab's icare() is by computing low rank factors of the Riccati
    %   solution, also in the form M = Z*Zᵀ.
    %
    %   This class is designed to facilitate handling the matrix M, but by
    %   storing only its square-root factor Z. The goal is to internally just
    %   store Z, but externally have something that looks and quacks like the
    %   matrix M. This is primarily done by overloading key methods so that the
    %   user doesn't have to worry about these details.
    %
    %   This class overloads the following methods so that a factoredMatrix
    %   behaves similarly to a regular matrix:
    %       - factoredMatrix: constructor, just pass in Z
    %       - mtimes: performs matrix-matrix multiplication as M*B = Z*(Z.'*B)
    %       - full: evaluate and return the full matrix (generally not recommended)
    %       - disp: display the matrix or its square-root factor
    %       - chol: returns the square root factor Z
    %       - vec: fake overload of vectorize
    %       - length: return n, the height of Z
    %       - size: M = Z*Zᵀ, so M is square n x n where n is height of Z
    %       - reshape: fake overload, only works for n x n and will just return M
    %       - transpose: M is symmetric, so just returns M
    properties
        Z
    end
    
    methods
        function obj = factoredMatrix(Z)
            obj.Z = Z;
        end
        
        function result = mtimes(obj, other)
            if isa(obj, 'factoredMatrix') && isa(other, 'factoredMatrix')
                % Both are factoredMatrix; not planning on using this
                result = obj.Z * (obj.Z.' * other.Z) * other.Z.';
            elseif isa(other, 'factoredMatrix') 
                % A*B where B is a factoredMatrix
                result = (obj * other.Z) * other.Z.';
            else 
                % A*B where A is a factoredMatrix
                result = obj.Z * (obj.Z.' * other);
            end
        end
        
        function result = full(obj)
            result = obj.Z * obj.Z.';
        end
        
        function disp(obj, dispFull)
            if nargin < 2
                dispFull = false;
            end
            
            if dispFull
                disp(full(obj))
            else
                builtin('disp', obj);
                fprintf("    Call disp(M,true) to display the full M matrix.\n\n")
            end
        end
        
        function result = chol(obj, triangle)
            if nargin < 2 % Note: my default is lower, but standard chol defaults to upper
                triangle = 'lower';
            end
            
            switch triangle
                case 'lower'
                    result = obj.Z;
                otherwise
                    result = obj.Z.';
            end
        end
        
        
        function result = vec(obj)
            % This is a fake overload; the result is not vec(obj) it is just obj.
            % In the PPR code, we use v2 = vec(V2); however, later on, it
            % is (should) always be reshaped again as V2. In general, in
            % Kronecker product computations, we should NEVER use the
            % vector, it should always end up reshaped as the matrix. So,
            % just to have the code easier to read so that we don't have to
            % always have an exception for v2, we will just fake it and
            % then have the reshaping operation handle it later.
            result = obj;
        end
        
        function result = length(obj)
            result = size(obj.Z, 1);
        end
        
        function result = size(obj)
            n = length(obj);
            result = [n, n];
        end
        
        function result = reshape(obj, dim1, dim2)
            n = length(obj);
            if any(dim1 == n) || any(dim2 == n)
                result = obj;
            else
                error("factoredMatrix: M should be a square matrix")
            end
        end
        
        function result = transpose(obj)
            % Assuming obj is symmetric
            result = obj;
        end
    end
    
end
