function [c] = kronMonomialSymmetrize(c, b, d, verbose)
%kronMonomialSymmetrize() symmetrizes Kronecker monomial coefficients.
%
%  Input Variables:
%     c - coefficients of the Kronecker polynomial (row vector of coefficients)
%     b - number of variables in the multinomial
%     d - degree of the multinomial terms (c should have b^d entries)
%
%  Returns: symmetrized coefficients of the multinomial
%
%  Rewritten to leverage the tensor toolbox:
%
%  Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB,
%  Version 3.5, www.tensortoolbox.org, February 25, 2023.
%
%  Part of the KroneckerTools repository.
%%
vec = @(X) X(:);

try
    c = vec(double(symmetrize(tensor(reshape(c, ones(1, d) * b)))));
catch
    error('Did you clone the tensor_toolbox repo and add it to path?')
end

end
