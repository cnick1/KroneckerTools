function [X] = lyap2c(A, E, X, TRANS)
%lyap2c Solves the reduced continuous-time generalized Lyapunov equation
%
%   Usage: X = lyap2c(A,E,Y,TRANS)
%
%   Inputs:
%       A      - coefficient matrix in triangular form (n x n)
%       E      - mass matrix in triangular form        (n x n)
%       Y      - symmetric right-hand-side matrix      (n x n)
%       TRANS  - whether to solve the transposed version
%
%   Output:
%       X      - solution to x = ℒ₂ᴱ(A)⁻¹vec(Y)
%
%   Background: Define the 2-way Lyapunov Matrix
%
%       ℒ₂ᴱ(A) = ( A ⊗ E  +  E ⊗ A )
%
%   Then the linear system ℒ₂ᴱ(A)x = b is the Kronecker form of the Lyapunov
%   equation, where x = vec(X) and b = vec(Y). For example,
%
%       ℒ₂ᴱ(A)x = b  ⟷  (A⊗E + E⊗A)vec(X) = vec(Y)  ⟷  AXE' + EXA' = Y
%
%   This function implements the generalized Bartels-Stewart algorithm of
%   Penzl [1], with computational complexity O(knᵏ⁺¹). The Bartels-Stewart
%   algorithm consists of 3 steps:
%       1) Transform to the triangular form
%       2) Solve the transformed system
%       3) Transform the solution back
%
%   This function specifically focuses on step 2, which is the main
%   challenge, since the other steps just involve transforming by Q and Z.
%   This function contains the less efficient but simpler implementation
%   based on the complex QZ decomposition, which ensures that A and E are
%   triangular. If TRANS=true, the transposed version is solved:
%
%       (A'⊗E' + E'⊗A')vec(X) = vec(Y)  ⟷  A'XE + E'XA = Y
%
%   This function replicates what Matlab's lyap() should be doing under the
%   hood. It is specifically based on SG03AY.f from SLICOT [2], which should
%   be what Matlab calls [3], the theory of which is presented in [1].
%
%   Authors: Nicholas Corbin, UC San Diego
%
%   References: [1] T. Penzl, “Numerical solution of generalized Lyapunov
%                equations,” Advances in Computational Mathematics, vol. 8,
%                no. 1/2, pp. 33–48, 1998, doi: 10.1023/a:1018979826766.
%               [2] https://github.com/SLICOT/SLICOT-Reference/blob/main/src/SG03AY.f
%               [3] https://www.slicot.org/matlab-toolboxes/basic-control/basic-control-performance-results
%
%   License: MIT
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%
N = size(A,1);

if TRANS
    % Solve Equation (1)
    for K = 1:N                                                       % go from first row to last
        % Copy elements of solution already known by symmetry
        if K > 1
            X(K,1:K-1) = X(1:K-1,K).';
        end

        for L = K:N                                        % go through columns from the left to the diagonal
            % Update right-hand sides (I)
            if L > 1 % No back substitution on the first system
                X(K:L, L) = X(K:L, L) ...
                    - A(K, K:L).' * ( X(K, 1:L-1) * E(1:L-1, L) ) ...
                    - E(K, K:L).' * ( X(K, 1:L-1) * A(1:L-1, L) );
            end

            % Solve small Sylvester equations of order at most (2,2)
            MAT = E(L, L) * A(K, K) + A(L, L) * E(K, K);
            RHS = X(K, L);

            RHS = MAT\RHS;

            % Assign solution values to X.
            X(K, L) = RHS;

            % Update right hand sides (II).
            if K < L
                X(K+1:L, L) = X(K+1:L, L) ...
                    - A(K, K+1:L).' * (X(K, L) * E(L, L)) ...
                    - E(K, K+1:L).' * (X(K, L) * A(L, L));
            end
        end
    end
else
    % Solve Equation (2)
    for L = N:-1:1                                                        % go from last column to first
        % Copy elements of solution already known by symmetry
        if L < N
            X(L+1:N,L) = X(L,L+1:N).';
        end

        % Inner Loop. Compute block X(KL,LL).
        for K = L:-1:1                                                    % go up rows from the diagonal to the top
            % Update right-hand sides (I)
            if K < N % No back substitution on the first system
                X(K, K:L) = X(K, K:L) ...
                    - ( A(K, K+1:N) * X(K+1:N, L) ) * E(K:L, L).' ...
                    - ( E(K, K+1:N) * X(K+1:N, L) ) * A(K:L, L).';
            end

            % Solve small Sylvester equations of order at most (2,2)
            MAT = E(L, L) * A(K, K) + A(L, L) * E(K, K);
            RHS = X(K, L);

            RHS = MAT\RHS;

            % Assign solution values to X.
            X(K, L) = RHS;

            % Update right hand sides (II).
            if K < L
                X(K, K:L-1) = X(K, K:L-1) ...
                    - A(K, K) * X(K, L) * E(K:L-1, L).' ...
                    - E(K, K) * X(K, L) * A(K:L-1, L).';
            end
        end
    end
end
end
