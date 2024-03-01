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
%  Rewritten based on the tensor toolbox:
%
%  Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB,
%  Version 3.5, www.tensortoolbox.org, February 25, 2023.
%
%  Part of the KroneckerTools repository.
%%
if nargin < 3
    k = log(length(c))/log(n);
end

% Construct matrix ind where each row is the multi-index for one element of X
idx = tt_ind2sub(ones(1, k) * n, (1:n ^ k)');

% Find reference index for every element in the tensor - this is to its
% index in the symmetrized tensor. This puts every element into a 'class'
% of entries that will be the same under symmetry.
classidx = sort(idx, 2); % Normalize to one permutation, i.e. reference element
mult = [1 cumprod(ones(1, k - 1) * n)]; % Form shifts
linclassidx = (classidx - 1) * mult' + 1; % Form vector that maps to the reference elements

try
    if isvector(c)
        c = symmetrizeHelper(c);
    else
        if size(c,1) == n
            for i=1:n
                c(i,:) = symmetrizeHelper(c(i,:));
            end

        elseif size(c,2) == n
            for i=1:n
                c(:,i) = symmetrizeHelper(c(:,i));
            end
        else
            try
                disp('Symmetrizing the rows of c')
                for i=1:size(c,1)
                    c(i,:) = symmetrizeHelper(c(i,:));
                end
            catch
                disp('Symmetrizing rows failed; trying symmetrizing the columns of c')
                for i=1:size(c,2)
                    c(:,i) = symmetrizeHelper(c(:,i));
                end
            end
        end
    end
catch
    error('kronMonomialSymmetrize: Something went wrong; did you clone the tensor_toolbox repo and add it to path?')
end

    function v = symmetrizeHelper(v)
        classsum = accumarray(linclassidx, v);
        classnum = accumarray(linclassidx, 1);

        avg = classsum ./ classnum;

        % Fill in each entry with its new symmetric version
        v = avg(linclassidx);
    end

end
