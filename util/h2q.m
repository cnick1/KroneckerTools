function [q] = h2q(h)
%h2q Compute coefficients for q(x) such that q(x) = h(x).' * h(x)
%   Given a cell array of the coefficients for h(x), compute the
% coefficients of q(x) such that q(x) = h(x).' * h(x). For example, if h(x)
% contains linear, quadratic terms, and cubic terms, a degree 4 term in q 
% will involve the cross terms between linear and cubic, and the cross
% terms between quadratic terms. 

% Create a vec function for readability
vec = @(X) X(:);

% Process inputs
n = size(h{1},2);
ell = length(h);

% Construct q(x) coefficients
for k = 2:2*ell
    q{k} = sparse(n^k,1);
    
    % TODO: use symmetry to cut in half
    for i = (k - ell):ell % would be 1:(k-1) but need to truncate only to h{} terms which exist
        j = k - i;
        q{k} = q{k} + vec(h{i}.' * h{j});
    end
end

end

