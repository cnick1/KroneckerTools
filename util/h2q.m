function [q] = h2q(h)
%h2q Compute coefficients for q(x) such that q(x) = h(x)ᵀ h(x)
%   This function is just a helper/utility function to compute the polynomial
%   coefficients of q(x) = h(x)ᵀ h(x) from the polynomial coefficients of h(x).
%   This problem arises when computing the balanced truncation energy functions.
%   For example, the observability energy function is defined as
%           Lₒ(x) = minᵤ ½∫ ||y(t)||² dt
%           s.t.    ẋ = f(x),   x(0) = x₀
%   Since y(t) = h(x), this leads to the nonlinear Lyapunov-type equation
%           0 = 𝜕ᵀV(x)/𝜕x f(x)  + ½ h(x)ᵀ h(x)
%   This is a special case of the HJB PDE that arises in nonlinear optimal
%   control. The more general optimal control problem is
%           minᵤ   J(x,u) = ½∫ xᵀ Q(x) x + uᵀ R(x) u dt
%           s.t.    ẋ = f(x) + g(x) u,   x(0) = x₀
%   which leads to the HJB PDE
%           0 = 𝜕ᵀV(x)/𝜕x f(x) - ½ 𝜕ᵀV(x)/𝜕x g(x) R⁻¹(x) gᵀ(x) 𝜕V(x)/𝜕x + ½ xᵀ Q(x) x
%   The function q(x) = xᵀ Q(x) x in this case is q(x) = h(x)ᵀ h(x).
%
%   For example, if the output function is cubic
%       h(x) = Cx + H₂(x⊗x) + H₃(x⊗x⊗x)
%   then q(x) will be degree 6
%       q(x) = xᵀCᵀCx + xᵀCᵀH₂(x⊗x) + (x⊗x)ᵀH₂ᵀCx + ... + (x⊗x⊗x)ᵀH₃ᵀH₃(x⊗x⊗x)
%   Notice the cross terms for the higher-order terms. This function does the
%   rearranging to get this in the form
%       q(x) = 1/2 ( q₂ᵀ(x⊗x) + q₃ᵀ(x⊗x⊗x) + ... + qᵈᵀ(...⊗x) )
%
%   As a matter of convenience, q₂ = vec(Q); the PPR function can handle either
%   q₂ or Q. Here, we use the factoredMatrix class to store Q=CᵀC in terms of C,
%   since when lyapchol() is called the matrix C is required.

% Create a vec function for readability
vec = @(X) X(:);

if ~iscell(h) & ismatrix(h)
    h = {h};
end

% Process inputs
n = size(h{1},2);
ell = length(h);

%% Construct q(x) coefficients

% First do Q2 = C.'*C separately to use factoredMatrix
q{2} = factoredMatrix(h{1}.');

% Now construct higher order terms, which contain cross terms
for k = 3:2*ell
    q{k} = sparse(n^k,1);
    
    % TODO: use symmetry to cut in half
    for i = max(1, k - ell):min(k-1, ell) % would be 1:(k-1) but need to truncate only to h{} terms which exist
        j = k - i;
        q{k} = q{k} + vec(h{i}.' * h{j});
    end
end

end

