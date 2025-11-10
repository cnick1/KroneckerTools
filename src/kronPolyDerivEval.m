function [x] = kronPolyDerivEval(c, z, degree)
%kronPolyDerivEval evaluates the derivative (gradient) of a Kronecker
%polynomial out to degree=degree, assuming the coefficients are symmetric!
%
%  Usage:
%     x = kronPolyDerivEval(c,z,degree);
%
%  Input Variables:
%     c - coefficients of the polynomial (cell array), assumed symmetric
%     z - value to calculate the polynomial at (vector)
%     degree - polynomial will be evaluated out to degree=degree
%              (default is the length of c)
%
%  Returns:  x = c{1} + 2*c{2}*z + ... degree*c{degree}*kron(...)
%
%  Author: Nick Corbin, UCSD
%  Heavily based on kronPolyEval by Jeff Borggaard, Virginia Tech
%
%  Licence: MIT
%
%  Part of the KroneckerTools repository.
%%
% Create a vec function for readability
vec = @(X) X(:);d = length(c);

if (nargin < 3)
    degree = d;
else
    if (d < degree)
        error('kronPolyEval: not enough entries in the coefficient cell array')
    end
end

if isa(c,'factoredValueArray')
    %% Transpose the coefficients
    c.ReducedValueCoefficients = cellfun(@transpose,c.ReducedValueCoefficients,'UniformOutput',false);
    
    %% Perform polynomial evalauation
    %  Special case if the linear term is an empty cell
    n = size(z, 1);
    if isempty(c{1})
        x = 0;
    else
        x = c{1}.';
    end
    
    % Quadratic term
    x  = x +  z.' * reshape(c.ReducedValueCoefficients{2},n,[]);
    
    z = c.Tinv*z; r = size(z, 1); zkm1 = z;
    for k=3:degree
        zkm1 = kron(zkm1, z);
        x = x + k * zkm1.' * reshape(c.ReducedValueCoefficients{k},r^(k-1),[]);
    end
    
else
    %% Perform polynomial evaluation
    %  Special case if the linear term is an empty cell
    n = size(z, 1);
    if isempty(c{1})
        x = 0;
    else
        x = c{1}.';
    end
    
    zkm1 = 1;
    for k = 2:degree
        %     % Naive
        %     zkm1 = kron(zkm1, z);
        %     x = x + k * c{k} * kron(speye(n), zkm1);
        
        %     % Using kron-vec identity (level-3 vs level-2 BLAS)
        zkm1 = kron(zkm1, z);
        x = x + k * zkm1.' * reshape(c{k},n^(k-1),[]);
        
        % tensor_toolbox method
        %     zkm1 = symktensor(z, k-1, 1, true);
        %     x = x + k * vec(double(zkm1)).' * reshape(c{k},n^(k-1),[]);
    end
end
end
