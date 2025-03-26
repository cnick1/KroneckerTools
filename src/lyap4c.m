function [X] = lyap4c(A, E, X, TRANS)
%lyap4c Solves the 4-way reduced continuous-time generalized Lyapunov equation
%
%   Usage: X = lyap4c(A,E,Y,TRANS)
%
%   Inputs:
%       A      - coefficient matrix in triangular form (n x n)
%       E      - mass matrix in triangular form        (n x n)
%       Y      - symmetric right-hand-side tensor      (n x n x n x n)
%       TRANS  - whether to solve the transposed version
%
%   Output:
%       X      - solution to x = ℒ₄ᴱ(A)⁻¹vec(Y)
%
%   Background: The linear system ℒ₄ᴱ(A)x = b is a tensor version of the Lyapunov
%   equation, where x = vec(X) and b = vec(Y). This function implements a generalized Bartels-Stewart algorithm that
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
if nargin < 4
    TRANS = false;
end

if TRANS
    % Solve Equation (1)
    for K = 1:N
        % Copy elements of solution already known by symmetry
        for L = K:N
            % Copy elements of solution already known by symmetry
            for M = L:N
                % Copy elements of solution already known by symmetry
                for NN = M:N
                    % ~~~~~~~~~~ Update right-hand sides I ~~~~~~~~~~
                    if NN > 1
                        X(K:NN,L,M,NN) = X(K:NN,L,M,NN) ...
                            - A(K,K:NN).' * (reshape(reshape(reshape(X(K,1:L,1:M,1:NN-1),[],NN-1) * E(1:NN-1,NN),[],M) * E(1:M,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(reshape(X(K,1:L,1:M,1:NN-1),[],NN-1) * A(1:NN-1,NN),[],M) * E(1:M,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(reshape(X(K,1:L,1:M,1:NN-1),[],NN-1) * E(1:NN-1,NN),[],M) * A(1:M,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(reshape(X(K,1:L,1:M,1:NN-1),[],NN-1) * E(1:NN-1,NN),[],M) * E(1:M,M),[],L) * A(1:L,L));
                    end
                    % ~~~~~~~~~~ Update right-hand sides II ~~~~~~~~~~~
                    if M > 1         % no back substitution on the first system
                        X(K:NN,L,M,NN) = X(K:NN,L,M,NN) ...
                            - A(K,K:NN).' * (reshape(reshape(X(K,1:L,1:M-1,NN) * E(NN,NN),[],M-1) * E(1:M-1,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(X(K,1:L,1:M-1,NN) * A(NN,NN),[],M-1) * E(1:M-1,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(X(K,1:L,1:M-1,NN) * E(NN,NN),[],M-1) * A(1:M-1,M),[],L) * E(1:L,L)) ...
                            - E(K,K:NN).' * (reshape(reshape(X(K,1:L,1:M-1,NN) * E(NN,NN),[],M-1) * E(1:M-1,M),[],L) * A(1:L,L));
                    end
                    % ~~~~~~~~~~ Update right-hand sides III ~~~~~~~~~~
                    if L > 1
                        X(K:NN,L,M,NN) = X(K:NN,L,M,NN) ...
                            - A(K,K:NN).' * (reshape(X(K,1:L-1,M,NN) * E(NN,NN) * E(M,M),[],L-1) * E(1:L-1,L)) ...
                            - E(K,K:NN).' * (reshape(X(K,1:L-1,M,NN) * A(NN,NN) * E(M,M),[],L-1) * E(1:L-1,L)) ...
                            - E(K,K:NN).' * (reshape(X(K,1:L-1,M,NN) * E(NN,NN) * A(M,M),[],L-1) * E(1:L-1,L)) ...
                            - E(K,K:NN).' * (reshape(X(K,1:L-1,M,NN) * E(NN,NN) * E(M,M),[],L-1) * A(1:L-1,L));
                    end
                    
                    
                    % Solve small Sylvester equation
                    MAT = E(NN,NN) * E(M,M) * E(L,L) * A(K,K) ...
                        + E(NN,NN) * E(M,M) * A(L,L) * E(K,K) ...
                        + E(NN,NN) * A(M,M) * E(L,L) * E(K,K) ...
                        + A(NN,NN) * E(M,M) * E(L,L) * E(K,K);
                    RHS = X(K, L, M, NN);
                    
                    % Assign solution values to X
                    X(K, L, M, NN) = MAT\RHS;
                    
                    X(K,L,NN,M) = X(K,L,M,NN);
                    X(K,M,L,NN) = X(K,L,M,NN);
                    X(K,M,NN,L) = X(K,L,M,NN);
                    X(K,NN,L,M) = X(K,L,M,NN);
                    X(K,NN,M,L) = X(K,L,M,NN);
                    X(L,K,M,NN) = X(K,L,M,NN);
                    X(L,K,NN,M) = X(K,L,M,NN);
                    X(L,M,K,NN) = X(K,L,M,NN);
                    X(L,M,NN,K) = X(K,L,M,NN);
                    X(L,NN,K,M) = X(K,L,M,NN);
                    X(L,NN,M,K) = X(K,L,M,NN);
                    X(M,K,L,NN) = X(K,L,M,NN);
                    X(M,K,NN,L) = X(K,L,M,NN);
                    X(M,L,K,NN) = X(K,L,M,NN);
                    X(M,L,NN,K) = X(K,L,M,NN);
                    X(M,NN,K,L) = X(K,L,M,NN);
                    X(M,NN,L,K) = X(K,L,M,NN);
                    X(NN,K,L,M) = X(K,L,M,NN);
                    X(NN,K,M,L) = X(K,L,M,NN);
                    X(NN,L,K,M) = X(K,L,M,NN);
                    X(NN,L,M,K) = X(K,L,M,NN);
                    X(NN,M,K,L) = X(K,L,M,NN);
                    X(NN,M,L,K) = X(K,L,M,NN);
                    
                    % ~~~~~~~~~~ Update right-hand sides IV ~~~~~~~~~
                    if K < NN
                        X(K+1:NN,L,M,NN) = X(K+1:NN,L,M,NN) ...
                            - A(K,K+1:NN).'*(X(K,L,M,NN)*E(NN,NN)*E(M,M)*E(L,L)) ...
                            - E(K,K+1:NN).'*(X(K,L,M,NN)*A(NN,NN)*E(M,M)*E(L,L)) ...
                            - E(K,K+1:NN).'*(X(K,L,M,NN)*E(NN,NN)*A(M,M)*E(L,L)) ...
                            - E(K,K+1:NN).'*(X(K,L,M,NN)*E(NN,NN)*E(M,M)*A(L,L));
                    end
                end
            end
        end
    end
else
    % Solve Equation (2)
    for L = N:-1:1
        % Symmetrize here
        for M = L:-1:1
            % Symmetrize here
            for NN = M:-1:1
                % Symmetrize here
                for K = NN:-1:1
                    % ~~~~~~~~~~ Update right-hand sides I ~~~~~~~~~~
                    if K < N
                        X(K,K:L,M,NN) = X(K,K:L,M,NN) ...
                            - ( reshape( reshape( A(K,K+1:N) * reshape(X(K+1:N,L,M:N,NN:N),N-K,[]) ,[],N-NN+1) * E(NN,NN:N).',[],N-M+1) * E(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K+1:N) * reshape(X(K+1:N,L,M:N,NN:N),N-K,[]) ,[],N-NN+1) * A(NN,NN:N).',[],N-M+1) * E(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K+1:N) * reshape(X(K+1:N,L,M:N,NN:N),N-K,[]) ,[],N-NN+1) * E(NN,NN:N).',[],N-M+1) * A(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K+1:N) * reshape(X(K+1:N,L,M:N,NN:N),N-K,[]) ,[],N-NN+1) * E(NN,NN:N).',[],N-M+1) * E(M,M:N).' ) * A(K:L,L).';
                    end
                    % ~~~~~~~~~~ Update right-hand sides II ~~~~~~~~~~~
                    if NN < N
                        X(K,K:L,M,NN) = X(K,K:L,M,NN) ...
                            - ( reshape( reshape( A(K,K) * X(K,L,M:N,NN+1:N),[],N-NN) * E(NN,NN+1:N).',[],N-M+1) * E(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K) * X(K,L,M:N,NN+1:N),[],N-NN) * A(NN,NN+1:N).',[],N-M+1) * E(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K) * X(K,L,M:N,NN+1:N),[],N-NN) * E(NN,NN+1:N).',[],N-M+1) * A(M,M:N).' ) * E(K:L,L).' ...
                            - ( reshape( reshape( E(K,K) * X(K,L,M:N,NN+1:N),[],N-NN) * E(NN,NN+1:N).',[],N-M+1) * E(M,M:N).' ) * A(K:L,L).';
                    end
                    % ~~~~~~~~~~ Update right-hand sides III ~~~~~~~~~~
                    if M < N
                        X(K,K:L,M,NN) = X(K,K:L,M,NN) ...
                            - ( reshape(A(K,K) * X(K,L,M+1:N,NN) * E(NN,NN).', [], N-M) * E(M,M+1:N).' ) * E(K:L,L).' ...
                            - ( reshape(E(K,K) * X(K,L,M+1:N,NN) * A(NN,NN).', [], N-M) * E(M,M+1:N).' ) * E(K:L,L).' ...
                            - ( reshape(E(K,K) * X(K,L,M+1:N,NN) * E(NN,NN).', [], N-M) * A(M,M+1:N).' ) * E(K:L,L).' ...
                            - ( reshape(E(K,K) * X(K,L,M+1:N,NN) * E(NN,NN).', [], N-M) * E(M,M+1:N).' ) * A(K:L,L).';
                    end
                    
                    % Solve small Sylvester equation
                    MAT = E(NN,NN) * E(M,M) * E(L,L) * A(K,K) ...
                        + E(NN,NN) * E(M,M) * A(L,L) * E(K,K) ...
                        + E(NN,NN) * A(M,M) * E(L,L) * E(K,K) ...
                        + A(NN,NN) * E(M,M) * E(L,L) * E(K,K);
                    RHS = X(K, L, M, NN);
                    
                    % Assign solution values to X
                    X(K, L, M, NN) = MAT\RHS;
                    
                    X(K,L,NN,M) = X(K,L,M,NN);
                    X(K,M,L,NN) = X(K,L,M,NN);
                    X(K,M,NN,L) = X(K,L,M,NN);
                    X(K,NN,L,M) = X(K,L,M,NN);
                    X(K,NN,M,L) = X(K,L,M,NN);
                    X(L,K,M,NN) = X(K,L,M,NN);
                    X(L,K,NN,M) = X(K,L,M,NN);
                    X(L,M,K,NN) = X(K,L,M,NN);
                    X(L,M,NN,K) = X(K,L,M,NN);
                    X(L,NN,K,M) = X(K,L,M,NN);
                    X(L,NN,M,K) = X(K,L,M,NN);
                    X(M,K,L,NN) = X(K,L,M,NN);
                    X(M,K,NN,L) = X(K,L,M,NN);
                    X(M,L,K,NN) = X(K,L,M,NN);
                    X(M,L,NN,K) = X(K,L,M,NN);
                    X(M,NN,K,L) = X(K,L,M,NN);
                    X(M,NN,L,K) = X(K,L,M,NN);
                    X(NN,K,L,M) = X(K,L,M,NN);
                    X(NN,K,M,L) = X(K,L,M,NN);
                    X(NN,L,K,M) = X(K,L,M,NN);
                    X(NN,L,M,K) = X(K,L,M,NN);
                    X(NN,M,K,L) = X(K,L,M,NN);
                    X(NN,M,L,K) = X(K,L,M,NN);
                    
                    % ~~~~~~~~~~ Update right-hand sides IV ~~~~~~~~~
                    if K < L
                        X(K,K:L-1,M,NN) = X(K,K:L-1,M,NN) ...
                            - (A(K,K) * X(K,L,M,NN) * E(NN,NN).' * E(M,M).') * E(K:L-1,L).' ...
                            - (E(K,K) * X(K,L,M,NN) * A(NN,NN).' * E(M,M).') * E(K:L-1,L).' ...
                            - (E(K,K) * X(K,L,M,NN) * E(NN,NN).' * A(M,M).') * E(K:L-1,L).' ...
                            - (E(K,K) * X(K,L,M,NN) * E(NN,NN).' * E(M,M).') * A(K:L-1,L).';
                    end
                end
            end
        end
    end
end
end
