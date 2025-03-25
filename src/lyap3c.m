function [X] = lyap3c(A, E, X, TRANS)
%lyap3c Solves the 3-way reduced continuous-time generalized Lyapunov equation
%
%   Usage: X = lyap3c(A,E,Y,TRANS)
%
%   Inputs:
%       A      - coefficient matrix in triangular form (n x n)
%       E      - mass matrix in triangular form        (n x n)
%       Y      - symmetric right-hand-side tensor      (n x n x n)
%       TRANS  - whether to solve the transposed version
%
%   Output:
%       X      - solution to x = ℒ₃ᴱ(A)⁻¹vec(Y)
%
%   Background: Define the 3-way Lyapunov Matrix
%
%   ℒ₃ᴱ(A) = ( A ⊗ E ⊗ E  +  E ⊗ A ⊗ E  +  E ⊗ E ⊗ A )
%
%   Then the linear system ℒ₃ᴱ(A)x = b is a tensor version of the Lyapunov
%   equation, where x = vec(X) and b = vec(Y). For example,
%
%       ℒ₃ᴱ(A)x = b  ⟷  (A⊗E⊗E + E⊗A⊗E + E⊗E⊗A)vec(X) = vec(Y)
%
%   This function implements a generalized Bartels-Stewart algorithm that
%   generalizes the method of Penzl [1], with computational complexity
%   O(knᵏ⁺¹) (I need to confirm this). The Bartels-Stewart algorithm
%   consists of 3 steps:
%       1) Transform to the triangular form
%       2) Solve the transformed system
%       3) Transform the solution back
%
%   This function specifically focuses on step 2, which is the main
%   challenge, since the other steps just involve transforming by Q and Z.
%   This function contains the less efficient but simpler implementation
%   based on the complex QZ decomposition, which ensures that A and E are
%   triangular.
%
%   Authors: Nicholas Corbin, UC San Diego
%
%   References: [1] T. Penzl, “Numerical solution of generalized Lyapunov
%                equations,” Advances in Computational Mathematics, vol. 8,
%                no. 1/2, pp. 33–48, 1998, doi: 10.1023/a:1018979826766.
%
%   License: MIT
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%
N = size(A,1);

if TRANS
    % Solve Equation (1)
    for K = 1:N
        % Copy elements of solution already known by symmetry
        if K > 1
            X(K,K,1:K-1) = permute(X(1:K-1,K,K),[3 2 1]);
            X(K,1:K-1,K) = permute(X(1:K-1,K,K),[3 2 1]);
        end

        for L = K:N
            % Copy elements of solution already known by symmetry
            if L > K
                X(K,L,1:L-1) = permute(X(K,1:L-1,L),[1 3 2]);
                X(L,K,1:L-1) = permute(X(K,1:L-1,L),[3 1 2]);
                X(1:L-1,K,L) = permute(X(K,1:L-1,L),[2 1 3]);
            end
            for M = L:N
                % ~~~~~~~~~~ Update right-hand sides I (eq. 44) ~~~~~~~~~~~
                % No back substitution on the first system
                if M > 1
                    X(K:M, L, M) = X(K:M, L, M) ...
                        - A(K, K:M)' * (reshape(reshape(X(K, 1:L, 1:M-1),[],M-1) * E(1:M-1, M),[],L) * E(1:L, L)) ...
                        - E(K, K:M)' * (reshape(reshape(X(K, 1:L, 1:M-1),[],M-1) * A(1:M-1, M),[],L) * E(1:L, L))...
                        - E(K, K:M)' * (reshape(reshape(X(K, 1:L, 1:M-1),[],M-1) * E(1:M-1, M),[],L) * A(1:L, L));
                end
                % ~~~~~~~~~~ Update right-hand sides II (eq. 45) ~~~~~~~~~~
                if L > 1
                    X(K:M, L, M) = X(K:M, L, M) ...
                        - A(K, K:M)' * ((X(K, 1:L-1, M) * E(M, M)) * E(1:L-1, L)) ...
                        - E(K, K:M)' * ((X(K, 1:L-1, M) * A(M, M)) * E(1:L-1, L)) ...
                        - E(K, K:M)' * ((X(K, 1:L-1, M) * E(M, M)) * A(1:L-1, L));
                end


                % Solve small Sylvester equation
                MAT = E(M, M) * E(L, L) * A(K, K) + E(M, M) * A(L, L) * E(K, K) + A(M, M) * E(L, L) * E(K, K);
                RHS = X(K, L, M);

                % Assign solution values to X
                X(K, L, M) = MAT\RHS;

                % ~~~~~~~~~~ Update right-hand sides III (eq. 46) ~~~~~~~~~
                if K < M
                    X(K+1:M, L, M) = X(K+1:M, L, M) ...
                        - A(K, K+1:M)' * ((X(K,L,M) * E(M, M)) * E(L, L)) ...
                        - E(K, K+1:M)' * ((X(K,L,M) * A(M, M)) * E(L, L))...
                        - E(K, K+1:M)' * ((X(K,L,M) * E(M, M)) * A(L, L));
                end
            end
        end
    end
else
    % Solve Equation (2)
    for L = N:-1:1
        % Copy elements of solution already known by symmetry
        for M = L:-1:1
            % Copy elements of solution already known by symmetry
            for K = M:-1:1
                % ~~~~~~~~~~ Update right-hand sides I (eq. 13) ~~~~~~~~~~~
                if K < N         % no back substitution on the first system
                    X(K, K:L, M) = X(K, K:L, M) ...
                        - ( reshape(A(K, K+1:N) * reshape(X(K+1:N, L, M:N),N-K,[]),[],N-M+1) * E(M, M:N)' ) * E(K:L, L)' ...
                        - ( reshape(E(K, K+1:N) * reshape(X(K+1:N, L, M:N),N-K,[]),[],N-M+1) * A(M, M:N)' ) * E(K:L, L)' ...
                        - ( reshape(E(K, K+1:N) * reshape(X(K+1:N, L, M:N),N-K,[]),[],N-M+1) * E(M, M:N)' ) * A(K:L, L)';
                end
                % ~~~~~~~~~~ Update right-hand sides II (eq. 14) ~~~~~~~~~~
                if M < N
                    X(K, K:L, M) = X(K, K:L, M) ...
                        - ( reshape(A(K, K) * X(K, L, M+1:N),[],N-M) * E(M, M+1:N)' ) * E(K:L, L)' ...
                        - ( reshape(E(K, K) * X(K, L, M+1:N),[],N-M) * A(M, M+1:N)' ) * E(K:L, L)' ...
                        - ( reshape(E(K, K) * X(K, L, M+1:N),[],N-M) * E(M, M+1:N)' ) * A(K:L, L)';
                end

                % Solve small Sylvester equation
                MAT = E(M, M) * E(L, L) * A(K, K) + E(M, M) * A(L, L) * E(K, K) + A(M, M) * E(L, L) * E(K, K);
                RHS = X(K, L, M);

                RHS = MAT\RHS;

                % Assign solution values to X.
                X(K, L, M) = RHS;

                X(K, M, L) = X(K, L, M);
                X(L, K, M) = X(K, L, M);
                X(L, M, K) = X(K, L, M);
                X(M, K, L) = X(K, L, M);
                X(M, L, K) = X(K, L, M);


                % ~~~~~~~~~~ Update right-hand sides III (eq. 15) ~~~~~~~~~
                if K < L
                    X(K, K:L-1, M) = X(K, K:L-1, M) ...
                        - ( A(K, K) * X(K, L, M) * E(M, M)' ) * E(K:L-1, L)' ...
                        - ( E(K, K) * X(K, L, M) * A(M, M)' ) * E(K:L-1, L)' ...
                        - ( E(K, K) * X(K, L, M) * E(M, M)' ) * A(K:L-1, L)';
                end
            end
        end
    end
end
end
