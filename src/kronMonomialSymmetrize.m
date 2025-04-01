function [c] = kronMonomialSymmetrize(c, n, k)
%kronMonomialSymmetrize Symmetrize a vectorized k-way tensor c of dimension nᵏ
%
%   Usage: c = kronMonomialSymmetrize(c, n, k)
%
%   Inputs:
%      c - vectorized tensor (coefficient of the Kronecker monomial)
%            (c can also be a matrix of dimension m x nᵏ or nᵏ x m,   
%              in which case the columns/rows will be symmetrized)
%      n - number of variables in the monomial
%      k - degree of the monomial terms (c should have nᵏ entries)
%
%   Output:
%      c - symmetrized result
%
%   Description: Let Cₖ be a k-way tensor of dimension n, i.e. 
%
%       Cₖ( n x n x ... x n )
%          |- k indices -|
%
%   For example, an n x n matrix is a 2-way tensor. The tensor C is
%   symmetric if you can permute any of the indices, e.g. Cᵢⱼ = Cⱼᵢ for a
%   2-way tensor i.e. matrix. Symmetry is important when describing
%   multivariate polynomials, e.g. xᵀ Q₁ x = xᵀ Q₂ x if Q₁ and Q₂ share the
%   same symmetrization, and xᵀ Q₁ x = 0 if and only if Q₁ is skew-symmetric.
%   Higher-order mononomials can be written with tensor products and tensor
%   coefficients, or by vectorizing the coefficients. So for example the
%   quadratic monomial xᵀ Q₂ x can be rewritten as q₂ᵀ(x ⊗ x), where
%   q₂ = vec(Q₂). It is very straightforward to write higher-order monomials
%   in this way, e.g. for cₖ = vec(Cₖ)
% 
%       cₖᵀ( x ⊗ ... ⊗ x ) 
%           |- k factors -|
%   
%   To symmetrize cₖ or Cₖ amounts to a gather/scatter-type operation,
%   where we collect all of the elements that are equivalent due to
%   symmetry, average them, and distribute the average to all of the
%   element locations. The functions tt_ind2sub() and tt_sub2ind() from the 
%   tensor toolbox [1] perform the operations of finding finding these 
%   indices. 
%
%
%   References: [1] Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox 
%               for MATLAB, Version 3.5, www.tensortoolbox.org, February 25, 2023.
%
%  Part of the KroneckerTools repository.
%%
if nargin < 3
    k = log(length(c))/log(n);
end

%% Set up symmetrization helper function
    % Get symmetrization indices/parameters once and reuse in the case of
    % symmetrizing many rows/columns

    % Construct matrix ind where each row is the multi-index for one element of X
    idx = tt_ind2sub(ones(1, k) * n, (1:n ^ k)');
    
    % Find reference index for every element in the tensor - this is to its
    % index in the symmetrized tensor. This puts every element into a 'class'
    % of entries that will be the same under symmetry.
    classidx = sort(idx, 2); % Normalize to one permutation, i.e. reference element
    mult = [1 cumprod(ones(1, k - 1) * n)]; % Form shifts
    linclassidx = (classidx - 1) * mult' + 1; % Form vector that maps to the reference elements
    classnum = accumarray(linclassidx, 1);

    function v = symmetrizeHelper(v)
        %symmetrizeHelper Performs the gather/scatter averaging operation
        %   Input/Output: v - vector of dimension nᵏ, symmetrized result
        classsum = accumarray(linclassidx, v);
        avg = classsum ./ classnum;

        % Fill in each entry with its new symmetric version
        v = avg(linclassidx);
    end


%% Perform actual symmetrization
try
    if isvector(c)
        c = symmetrizeHelper(c);
    else
        if size(c,2) == n^k                      % symmetrize the rows of c
                for i=1:size(c,1)
                    c(i,:) = symmetrizeHelper(c(i,:));
                end
        elseif size(c,1) == n^k               % symmetrize the columns of c
                for i=1:size(c,2)
                    c(:,i) = symmetrizeHelper(c(:,i));
                end
        else
            error('kronMonomialSymmetrize: Dimension mismatch, either the columns or rows of c have to have dimension nᵏ')
        end
    end
catch
    error('kronMonomialSymmetrize: Something went wrong; did you clone the tensor_toolbox repo and add it to path?')
end

end

