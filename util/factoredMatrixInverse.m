classdef factoredMatrixInverse
    %FACTOREDMATRIXINVERSE An n x n matrix M stored in terms of its inverse square-root factors M = Z*Zᵀ = (Z⁻¹*Z⁻ᵀ)⁻¹.
    %   See the documentation for factoredMatrix first. That class is set
    %   up inspired by the observability energy, which for a linear system
    %   is given by the observability Gramian. The Gramian can be computed
    %   in terms of its Cholesky factor by lyapchol(), which gives
    %       W₂ = "Q" = L*Lᵀ
    %   and the observability energy for a nonlinear system is then expanded as
    %       Lₒ(x) = ½ ( xᵀ Q x + w₃ᵀ(x⊗x⊗x) + ... + wᵈᵀ(...⊗x) )
    %   The controllability energy instead is given by the INVERSE of the
    %   controllability Gramian.
    %       Lᶜ(x) = ½ ( xᵀ P⁻¹ x + v₃ᵀ(x⊗x⊗x) + ... + vᵈᵀ(...⊗x) )
    %   The Gramian can be computed in terms of its Cholesky factor by
    %   lyapchol(), which gives
    %       V₂ = "P⁻¹" = (R⁻¹*R⁻ᵀ)⁻¹ = R*Rᵀ
    %   In the end, what we actually need are not L and R, but L and R⁻¹.
    %   So instead of inverting twice, which is illconditioned², we can
    %   just deal directly with "R⁻¹", which is what lyapchol() gives as
    %   the square-root of the controllability Gramian.
    
    %   This class is designed to facilitate handling the matrix V₂, but by
    %   storing only its inverse square-root factor R⁻¹ and never actually
    %   performing any inversions. V₂ is used in two ways:
    %       1) for computing v₃, v₄, etc.
    %       2) for computing the first term in the balancing transformation
    %   Previously, an inverse operation was used in both 1) and 2). Task 1)
    %   when computing v₃, v₄, etc. does require using P⁻¹, whereas Task 2) can
    %   actually be done entirely without it and only using R⁻¹, which is the
    %   thing given by lyapchol().
    %
    %   For Task 1), previously, by computing V₂ = P⁻¹ ("yikes") we avoided
    %   having to repeatedly solve linear systems and instead just did it once
    %   (n solves) to get V₂ = P⁻¹ and then use matrix multiplication. To use the
    %   square-root factors instead, we have to do something like
    %       V₂*B = (R⁻¹*R⁻ᵀ)⁻¹ * B
    %            = (R⁻¹*R⁻ᵀ) \ B
    %            = R⁻¹\(R⁻ᵀ\B)
    %            = Rinv\(Rinv.'\B)
    %   which requires TWO linear solves every time we try to multiply V₂. Both
    %   matrix multiplication and linear solves are of complexity O(n³), so for
    %   our computations which are at minimum O(n⁴) the difference in
    %   computational speed is negligible. However, especially in the case of
    %   small singular values, inverting is going to be less accurate than
    %   solving the linear systems each time, so the improvement in accuracy is
    %   going to be worth it. This is especially important when the matrix is
    %   poorly conditioned (non-minimal or close to non-minimal), which is the
    %   whole point.
    %
    %   So, the overloaded mtimes() function is especially important for Task
    %   1). The overloaded cholinv() instead, which returns Zinv, is especially
    %   important for Task 2).
    %
    %   From here on, I'll revert to the notation M = (Z⁻¹*Z⁻ᵀ)⁻¹ to be
    %   consistent with the general class as in factoredMatrix. This class is
    %   designed to facilitate handling the matrix M, but by storing only its
    %   inverse square-root factor Z⁻¹ and never actually performing any
    %   inversions. The goal is to internally just store Zinv, but externally
    %   have something that looks and quacks like the matrix M. This is
    %   primarily done by overloading key methods so that the user doesn't have
    %   to worry about these details.
    %
    %   This class overloads the following methods so that a factoredMatrix
    %   behaves similarly to a regular matrix:
    %       - factoredMatrix: constructor, just pass in Zinv
    %       - mtimes: perform matrix multiplication by appropriate mldivide operations
    %       - full: evaluate and return the full matrix (generally not recommended)
    %       - inv: return M⁻¹ = (Z⁻¹*Z⁻ᵀ) (would correspond to full Gramian P)
    %       - disp: display the matrix or its square-root factor
    %       - cholinv: returns the inverse square-root factor Zinv
    %       - vec: fake overload of vectorize
    %       - length: return n, the height of Zinv
    %       - size: M is square n x n where n is height of Zinv
    %       - reshape: fake overload, only works for n x n and will just return M
    %       - transpose: M is symmetric, so just returns M
    properties
        Zinv
    end
    
    methods
        function obj = factoredMatrixInverse(Zinv)
            obj.Zinv = Zinv;
        end
        
        function result = mtimes(obj, other)
            if isa(obj, 'factoredMatrixInverse') && isa(other, 'factoredMatrixInverse')
                % Both are factoredMatrixInverse; not planning on using this
                error("factoredMatrixInverse: mtimes not currently implemented when both are factoredMatrixInverse")
            elseif isa(other, 'factoredMatrixInverse')
                % A*B where B is a factoredMatrixInverse
                %   A*B = A*(Z⁻¹*Z⁻ᵀ)⁻¹
                %       = A/(Z⁻¹*Z⁻ᵀ)
                %       = (A/Z⁻¹)/Z⁻ᵀ
                %       = (A/Zinv)/Zinv.'
                result = (obj / other.Zinv) / other.Zinv.';
            else
                % A*B where A is a factoredMatrixInverse
                %   A*B = (Z⁻¹*Z⁻ᵀ)⁻¹ * B
                %       = (Z⁻¹*Z⁻ᵀ)\B
                %       = Z⁻¹\(Z⁻ᵀ\B)
                %       = Zinv\(Zinv.'\B)
                result = obj.Zinv \ (obj.Zinv.' \ other);
            end
        end
        
        function result = full(obj)
            % Don't recommend ever calling this
            result = inv(obj.Zinv * obj.Zinv.');
        end
        
        function result = inv(obj)
            result = obj.Zinv * obj.Zinv.';
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
        
        function result = cholinv(obj)
            result = obj.Zinv;
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
            result = size(obj.Zinv, 1);
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
                error("factoredMatrixInverse: M should be a square matrix")
            end
        end
        
        function result = transpose(obj)
            % Assuming obj is symmetric
            result = obj;
        end
    end
    
end
