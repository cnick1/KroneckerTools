function [Tv] = calTTv(T, m, k, v)
%calTTv Calculates the product 𝓣ₘ,ₖᵀ v ("calligraphic T transpose times v")
%  𝓣ₘ,ₖ denotes all unique tensor products with m terms and nᵏ columns
%
%  Usage:  Tv = calTTv(T,m,k,v)
%
%   Inputs:
%       T  - a cell array of transformation coefficients
%       m  - number of terms in the tensor product; v has dimension n^m
%       k  - the tensor products have nᵏ columns; output vector Tv has dimension n^k
%       v  - a matrix with n^m rows (or a vector of dimension n^m)
%
%   Output:
%       Tv - the result of the product "calligraphic T transpose times v"
%
%   Background: When applying a polynomial transformation to a polynomial
%   energy function, repeated products appear such as
%
%                   (T₁⊗T₂⊗... ⊗Tₘ)ᵀ v
%
%   We introduce the notation 𝓣 = (T₁⊗T₂⊗... ⊗Tₘ); more
%   specifically, 𝓣 is defined with two subscript indices: m and k.
%       - m denotes the number of terms in the Kronecker products; it is
%         also related to the dimension of the input vector v
%       - k is related to the dimension of the resultant vector Tv
%
%   Examples include
%       - 𝓣ₖ,ₖ  = T₁⊗T₁⊗...⊗T₁    (one term, k factors)
%       - 𝓣₃,₄ = T₁⊗T₁⊗T₂ + T₁⊗T₂⊗T₁ + T₂⊗T₁⊗T₁
%
%   Hence "𝓣ₘ,ₖ denotes all unique tensor products with m terms and
%   n^k columns" [1]. In this function, we evaluate the product 𝓣ₘ,ₖᵀ v
%   efficiently using the kronecker-vec identity recursively with the
%   function kroneckerRight. There may be improvements to be made in
%   kroneckerRight, e.g. to avoid transposition of large matrices.
%
%   Author: Rewritten by Nick Corbin, UCSD, based on code by Jeff Borggaard, VT
%
%   License: MIT

%   Reference: [1] B. Kramer, S. Gugercin, and J. Borggaard, “Nonlinear
%              balanced truncation: Part 2—model reduction on manifolds,”
%              arXiv, Feb. 2023. doi: 10.48550/ARXIV.2302.02036
%
%  Part of the NLbalancing repository.
%%

if m == 1
    Tv = T{k}.'*v;
    return
end

% Get a list of indices
[indexSet, mult] = findCombinations(m, k);

nTerms = size(indexSet, 1);

n = size(T{1}, 2);

Tv = zeros(n ^ k, size(v,2));
for i = 1:nTerms
    Tv = Tv + mult(i) * kronMonomialSymmetrize(kroneckerRight(v.', T(indexSet(i, :))).', n, k); % Can also symmetrize each thing
end

end

function [combinations, multiplicities] = findCombinations(m, k)
%findCombinations  Returns all combinations of m natural numbers that sum to k.

combinations = [];
multiplicities = [];
findCombinationsHelper(m, k, zeros(1, m), 1, 1);

    function findCombinationsHelper(m, k, combination, index, start)
        if index > m
            if k == 0
                combinations = [combinations; combination];
                multiplicities = [multiplicities; computeMultiplicity(combination)];
            end
        else
            for i = start:k
                combination(index) = i;
                findCombinationsHelper(m, k - i, combination, index + 1, i);
            end
        end
    end
end

function multiplicity = computeMultiplicity(comb)
n = length(comb);
multiplicity = factorial(n);
for i = unique(comb)
    count = sum(comb == i);
    multiplicity = multiplicity / factorial(count);
end
end

