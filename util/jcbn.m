function J = jcbn(F, x, symmetrize)
%jcbn Return the Jacobian J(x) = ∂f(x)/∂x of the function f(x) evaluated at x.
%
%   Usage:  J = jcbn(F, x)
%
%   Inputs:    F - cell array containing function coefficients
%              x - point at which to evaluate the Jacobian
%     symmetrize - boolean, symmetrize rows of F if they aren't symmetric
%
%   Description: The Jacobian J(x) is the matrix giving the partial derivatives
%   ∂f(x)/∂x of the function f(x). The function of interest is a polynomial
%
%              f(x) = F₁x + F₂(x⊗x) + ... + Fd(x...⊗x)
%
%   where the rows of the function coefficients are symmetrized. Then,
%   the Jacobian is given by
%
%       J(x) = ∂f(x)/∂x = F₁ + 2F₂(I⊗x) + ... + d Fd(I...⊗x)
%
%   Evaluating this explicitly is expensive, so this function uses the
%   Kronecker-vec identity to do this recursively and in a more efficient
%   manner.
%
%   One use case for this function is for computing the Jacobian of a
%   transformation. For example, the Jacobian J(z̄) of the nonlinear balancing
%   transformation x = ̅Φ(z̄) can be computed using this function if a
%   polynomial approximation to the transformation is known.
%
%   References: [1]
%
%   Part of the KroneckerTools repository.
%%
if nargin < 3
    symmetrize = false;
end

n = size(x, 1);

if symmetrize % I'm not sure if this is ever used, could possibly remove
    if isempty(F{1})
        J = 0;
    else
        J = 1/2*(F{1}+F{1}.');
    end
    
    xkm1 = 1;
    for k = 2:length(F)
        % Using kron-vec identity
        xkm1 = kron(xkm1, x);
        % Need to iterate over n rows to apply kron-vec identity
        for j = 1:n
            J(j,:) = J(j,:) + k * xkm1.' * reshape(kronMonomialSymmetrize(F{k}(j,:),n,k),n^(k-1),[]);
        end
    end
else
    if isempty(F{1})
        J = 0;
    else
        J = F{1};
    end
    
    xkm1 = 1;
    for k = 2:length(F)
        % Using kron-vec identity
        xkm1 = kron(xkm1, x);
        % Need to iterate over n rows to apply kron-vec identity
        for j = 1:n
            J(j,:) = J(j,:) + k * xkm1.' * reshape(F{k}(j,:),n^(k-1),[]);
        end
    end
end
end