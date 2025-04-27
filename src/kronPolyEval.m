function [FofX] = kronPolyEval(f,x,d,sprse)
%kronPolyEval Evaluate a Kronecker polynomial f(x) to degree d.
%
%   Usage:     FofX = kronPolyEval(f,x,d);
%
%   Input Variables:
%          f - cell array containing function coefficients
%          x - point at which to evaluate the Jacobian (vector)
%          d - polynomial will be evaluated out to degree=d
%              (default is the length of f)
%      sprse - optional; exploit sparsity if it is present
%              (defaults to true; permits NOT exploiting
%               sparsity if necessary for some reason)
%
%   Description: The function f(x) of interest is a polynomial
%
%              f(x) = F₁x + F₂(x⊗x) + ... + Fd(x...⊗x)
%
%   This is evaluated efficiently by recycling (x⊗x) and just adding one
%   more factor of x at a time. Kronecker polynomials are used throughout
%   our work: we represent the dynamics, cost function, value function, and
%   optimal control as Kronecker polynomials. We can also represent coordinate
%   transformations, energy functions, etc. This function kronPolyEval() can
%   be used to evaluate all of these, which leads to some special cases:
%       - Sometimes we may wish to evaluate only out to a certain degree,
%         e.g. if you have a polynomial feedback law but just want the
%         first term. The third input argument d permits this.
%       - Some function, like the value function, do not have a linear
%         component. If f{1} = [], it will be skipped.
%       - If reduction is used, we may wish to evaluate the value function
%         or gain using the reduced-order x≈Txᵣ just for the higher terms:
%            V(x) =    1/2 ( v₂ᵀ(x⊗x) + v₃ᵣᵀ(xᵣ⊗xᵣ⊗xᵣ) + ... +   vᵣᵈᵀ(...⊗xᵣ) )
%            u(x) = K₁ x + K₂ᵣ(xᵣ⊗xᵣ) +  K₃ᵣ(xᵣ⊗xᵣ⊗xᵣ) + ... +  Kᵣᵈ⁻¹(...⊗xᵣ) )
%         This function can handle these cases using the factoredGainArray class
%         and factoredValueArray class, which basically pass along the reduced
%         coefficients and the reduction basis T.
%       - Sparse coefficients can be handled; here we have a general-purpose
%         implementation, but in cases where performance is critical and
%         repeated evaluation of the SAME polynomial are performed,
%         additional special case speed-ups can be made that give MAJOR
%         improvements. If performance is critical, my current approach is 
%         to make a local function sparseKronPolyEval() to keep the code 
%         simple and make any other speedups I can. See runExample29() in 
%         the PPR repository for an example. 
%
%   Author: Rewritten by Nick Corbin, UCSD
%           Based on original version by Jeff Borggaard, Virginia Tech
%
%   License: MIT
%
%   Part of the KroneckerTools repository.
%%
lf = length(f);
if nargin < 4
    sprse = true;
    if nargin < 3
        d = lf;
    end
end
if lf < d
    error('kronPolyEval: not enough entries in the coefficient cell array')
end
if d == 0
    FofX = 0;
    return
end

if isa(f,'factoredGainArray')
    %% Evaluate polynomial feedback law
    %  u(x) = K₁ x + K₂ᵣ(xᵣ⊗xᵣ) +  K₃ᵣ(xᵣ⊗xᵣ⊗xᵣ) + ... +  Kᵣᵈ⁻¹(...⊗xᵣ) )
    % using factoredGainArray class and reduced-order xᵣ
    
    % Evaluate linear term (can't be empty for u(x))
    FofX = f{1}*x;
    
    % Get reduced-order state xᵣ and evaluate higher-degree terms
    x = f.Tinv*x; xk = x;
    for k=2:d
        xk = kron(xk,x);
        FofX = FofX + f.ReducedGains{k}*xk;
    end
elseif isa(f,'factoredValueArray')
    %% Evaluate polynomial value function
    %  V(x) = 1/2 ( v₂ᵀ(x⊗x) + v₃ᵣᵀ(xᵣ⊗xᵣ⊗xᵣ) + ... + vᵣᵈᵀ(...⊗xᵣ) )
    % using factoredValueArray class and reduced-order xᵣ
    
    % Transpose the coefficients v₂ᵀ,...,vᵣᵈᵀ
    f.ReducedValueCoefficients = cellfun(@transpose,f.ReducedValueCoefficients,'UniformOutput',false);
    
    % First term in V(x) is quadratic term
    FofX = f.ReducedValueCoefficients{2}*kron(x,x);
    
    % Get reduced-order state xᵣ and evaluate higher-degree terms
    x = f.Tinv*x; xk = kron(x,x);
    for k=3:d
        xk = kron(xk,x);
        FofX = FofX + f.ReducedValueCoefficients{k}*xk;
    end
elseif sprse && issparse(f{end})
    %% Evaluate sparse Kronecker polynomial
    % Note: in the case of REPEATED evaluation of THE SAME sparse
    % Kronecker polynomial, a MAJOR improvement can be obtained by using
    % persistent variables to find and store the sparsity pattern of the
    % coefficients. If performance is that critical,  my current approach is 
    % making a local function sparseKronPolyEval() to keep the code simple
    % and make any other speedups I can. See runExample29() in the PPR
    % repository for an example. This is just a general case without all of
    % the possible special cases speed-ups that are possible.
    n = length(x);
    
    % Assume transposition not needed, and assume linear term not empty
    FofX = f{1}*x;
    
    % Evaluate higher-degree terms successively, avoiding forming kron(x,x,...,x) (since it is expensive and only a few entries are needed)
    for k=2:d
        [Fi, Fj, Fv] = find(f{k});
        inds = cell(1, k); % Preallocate cell array for k indices
        [inds{:}] = ind2sub(repmat(n, 1, k), Fj);
        
        % Efficient sparse evaluation of f{k}*(x⊗...⊗x)
        xprod = ones(size(Fj));
        for p = 1:k
            xprod = xprod .* x(inds{p});
        end
        
        FofX = FofX + accumarray(Fi, Fv .* xprod, size(x));
    end
else
    %% Evaluate standard Kronecker polynomial
    %% Transpose the coefficients if required
    % Check dimensions of the last entry in f to see which dimension is the
    % appropriate size. To multiply, the second dimension  of c{k} has to be
    % length(x)^k; instead of checking all of them, just check the last
    % entry, with k=lf=length(f) already having been defined.
    [nRows, nCols] = size(f{lf});
    
    if (length(x)^lf == nCols)
        % No need to transpose
    elseif (length(x)^lf == nRows)
        % Need to transpose, otherwise dimensions will not work
        f = cellfun(@transpose,f,'UniformOutput',false);
        %       warning('Coefficient needed to be transposed; consider transposing to speed up.')
    else
        % Probably won't ever happen but good contingency
        % warning('Dimensions of polynomial coefficients are not consistent')
    end
    
    %% Perform polynomial evaluation
    % Evaluate linear term; handle special case if the linear term is an empty cell
    if isempty(f{1})
        n = size(f{2},1);
        FofX = zeros(n,1);
    else
        FofX = f{1}*x;
    end
    
    % Evaluate higher-degree terms successively
    xk = x;
    for k=2:d
        xk = kron(xk,x);
        FofX  = FofX + f{k}*xk;
    end
end

end
