function [x] = kronPolyDerivEval(c, z, degree)
%kronPolyDerivEval evaluates the derivative (gradient) of a Kronecker polynomial out to degree=degree
%
%  Usage:
%     x = kronPolyDerivEval(c,z,degree);
%
%  Input Variables:
%     c - coefficients of the polynomial (cell array)
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

d = length(c);
if (nargin < 3)
    degree = d;
else
    if (d < degree)
        error('kronPolyEval: not enough entries in the coefficient cell array')
    end
end

%% Transpose the coefficients if required
% Check dimensions of the last entry in c to see which dimension is the
% appropriate size. To multiply, the second dimension  of c{k} has to be
% length(z)^k; instead of checking all of them, just check the last
% entry, with k=d=length(c) already having been defined.
[nRows, nCols] = size(c{d});

if length(z) ^ d == nCols
    % No need to transpose
elseif length(z) ^ d == nRows
    % Need to transpose, otherwise dimensions will not work
    c = cellfun(@transpose, c, 'UniformOutput', false);
else
    % Probably won't ever happen but good contingency
    warning('Dimensions of polynomial coefficients are not consistent')
end

%% Perform polynomial evalauation
%  Special case if the linear term is an empty cell
n = size(z, 1);
if isempty(c{1})
    x = 0;
else
    x = c{1};
end

zkm1 = 1;
for k = 2:degree
    zkm1 = kron(zkm1, z);
    x = x + k * c{k} * kron(speye(n), zkm1);
end

end
