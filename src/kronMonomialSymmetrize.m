function [c] = kronMonomialSymmetrize(c, n, k)
%kronMonomialSymmetrize Symmetrizes Kronecker monomial coefficients.
%
%   Usage: c = kronMonomialSymmetrize(c, n, k)
%
%   Inputs:
%      c - coefficient of the Kronecker monomial
%      n - number of variables in the monomial
%      k - degree of the monomial terms (c should have n^k entries)
%
%   Output:
%      c - symmetrized coefficient of the monomial
%
%  Rewritten to leverage the tensor toolbox:
%
%  Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB,
%  Version 3.5, www.tensortoolbox.org, February 25, 2023.
%
%  Part of the KroneckerTools repository.
%%
vec = @(X) X(:);

if nargin < 3
    k = log(length(c))/log(n);
end

if isvector(c)
    try
        %     c = vec(double(symmetrize(tensor(reshape(c, ones(1, k) * n)))));
        c = vec(symmetrize(tensor(reshape(c, ones(1, k) * n))));
    catch
        try
            c = vec(symmetrize(tensor(reshape(full(c), ones(1, k) * n))));
            disp("kronMonomialSymmetrize: had to convert to full")
        catch
            error('Did you clone the tensor_toolbox repo and add it to path?')
        end
    end
else
    if size(c,1) == n
        try
            for i=1:n
                c(i,:) = vec(symmetrize(tensor(reshape(c(i,:), ones(1, k) * n))));
            end
        catch
            try
                for i=1:n
                    c(i,:) = vec(symmetrize(tensor(reshape(full(c(i,:)), ones(1, k) * n))));
                end
                disp("kronMonomialSymmetrize: had to convert to full")
            catch
                error('Did you clone the tensor_toolbox repo and add it to path?')
            end
        end
    else
        try
            for i=1:n
                c(:,i) = vec(symmetrize(tensor(reshape(c(:,i), ones(1, k) * n))));
            end
        catch
            try
                for i=1:n
                    c(:,i) = vec(symmetrize(tensor(reshape(full(c(:,i)), ones(1, k) * n))));
                end
                disp("kronMonomialSymmetrize: had to convert to full")
            catch
                error('Did you clone the tensor_toolbox repo and add it to path?')
            end
        end
    end
end

end
