function LkMv = LyapProduct(M,v,k,E)
%LyapProduct Computes the k-way Lyapunov product ℒₖ(M) v
%
%   Usage:  LkMv = LyapProduct(M,v,k,E);
%
%   Inputs:
%       M       - matrix M, the argument of ℒₖ(M);  dimensions (m  x n)
%       v       - vector to multiply with: ℒₖ(M) v; dimensions (nᵏ x 1)
%       k       - degree of the Lyapunov product; determines how many
%                 Kronecker products are in each term of ℒₖ(M)
%       E       - mass matrix; dimensions (n x n)
%                   (optional, defaults to E = I)
%
%   Output:
%       LkMv     - result of the product ℒₖ(M) v
%
%   Description: The k-way Lyapunov matrix is defined as the mnᵏ⁻¹ x nᵏ matrix
%
%            |-----------------------------------k terms-----------------|
%   ℒₖ(M) = ( M⊗I⊗ ... ⊗I  +  I⊗M⊗ ... ⊗I  +  ...  +  I⊗I⊗ ... ⊗M )
%            |--k factors--|    |--k factors--|    ...    |--k factors--|
%
%   This matrix is very large and expensive to compute due to the storage
%   requirements; however, all of the information is contained in the
%   argument M, and typically the matrix ℒₖ(M) is not needed, but rather
%   its product with a vector ℒₖ(M) v is of interest. In this case, the
%   Kronecker product can be leveraged to efficiently compute this by
%   reshaping and permuting things to instead use matrix multiplication.
%   This arises in the Polynomial-Polynomial Regulator problem (PPR), a
%   generalization of the Linear-Quadratic, Quadratic-Quadratic, and
%   Polynomial-Quadratic Regulators (LQR, QQR, PQR). Details are provided
%   in [1-3].
%
%   Optionally, the function can be passed a mass matrix E that replaces
%   the identity matrices, i.e. to define
%
%       ℒₖᴱ(M) = ( M⊗E⊗...⊗E  +  E⊗M⊗...⊗E  + ... +  E⊗E⊗...⊗M )
%
%   Authors: Jeff Borggaard, Virginia Tech (Original author)
%            Nicholas Corbin, UC San Diego (Update to include E matrix)
%
%   Reference: [1] J. Borggaard and L. Zietsman, “The quadratic-quadratic
%               regulator problem: approximating feedback controls for
%               quadratic-in-state nonlinear systems,” in 2020 American
%               Control Conference (ACC), Jul. 2020, pp. 818–823. doi:
%               10.23919/ACC45564.2020.9147286
%              [2] J. Borggaard and L. Zietsman, “On approximating
%               polynomial-quadratic regulator problems,” IFAC-PapersOnLine,
%               vol. 54, no. 9, pp. 329–334, 2021, doi:
%               10.1016/j.ifacol.2021.06.090.
%              [3] N. A. Corbin and B. Kramer, "Computing solutions to the
%               polynomial-polynomial regulator problem,” in 2024 63rd IEEE
%               Conference on Decision and Control, Dec. 2024
%              [4] E. G. Al’brekht, “On the optimal stabilization of
%               nonlinear systems," Journal of Applied Mathematics and
%               Mechanics, vol. 25, no. 5, pp. 1254–1266, Jan. 1961, doi:
%               10.1016/0021-8928(61)90005-3
%              [5] D. L. Lukes, “Optimal regulation of nonlinear dynamical
%               systems,” SIAM Journal on Control, vol. 7, no. 1, pp.
%               75–100, Feb. 1969, doi: 10.1137/0307007
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%
vec = @(X) X(:);
[m,n] = size(M);

if ( size(v,1) ~= n^k ); error('The dimensions of v do not match the degree of the k-way Lyapunov matrix'); end % Right now, we are assuming v is a single column
if ( k<2 ); error('k must be >= 2'); end

if nargin < 4 || isempty(E) % no mass matrix E
    % Compute the first and last terms in the sum
    LkMv =        vec( reshape(v,n^(k-1),n) * M.' );
    LkMv = LkMv + vec( M * reshape(v,n,n^(k-1)) );
    
    % Loop to compute the remaining permutations
    for l=1:k-2
        V1 = reshape(v, n^(k-l), n^l);
        
        mat = zeros(m*n^(k-l-1), n^l);
        for i=1:n^l
            vi = V1(:, i);
            mat(:, i) = vec( reshape(vi,n^(k-l-1),n) * M.' );
        end
        LkMv = LkMv + vec(mat);
    end
else % with mass matrix E
    % Compute the first and last terms in the sum
    LkMv =        vec(  kroneckerLeft(E, reshape(v,n^(k-1),n) * M.') );
    LkMv = LkMv + vec( kroneckerRight(M * reshape(v,n,n^(k-1)), E.')   );
    
    % Loop to compute the remaining permutations
    for l=1:k-2
        V1 = reshape(v,n^(k-l),n^l);
        
        mat = zeros(m*n^(k-l-1),n^l);
        for i=1:n^l
            vi = V1(:,i);
            mat(:,i) = vec( kroneckerLeft(E, reshape(vi,n^(k-l-1),n)*M.') );
        end
        LkMv = LkMv + vec( kroneckerRight(mat, E.') );
    end
    
end

LkMv = LkMv(:); % redundant?
end

