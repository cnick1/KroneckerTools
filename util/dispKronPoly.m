function dispKronPoly(f,nvp)
%dispKronPoly Display a Kronecker Polynomial, rounding small quantities to zero
%
%   Usage:     dispKronPoly(f)
%
%   Input Variables:
%          f - cell array containing function coefficients
%
%   Additional name/value pair options:
%               n - dimension of the state variable x
%        variable - the symbol to use when printing, e.g. x, z, etc.
%          thresh - truncation threshold (default 1e-7)
%          degree - polynomial will be evaluated out to degree=d
%                   (default is the length of f)
%
%   Description: The function f(x) of interest is a polynomial
%
%              f(x) = F₁x + F₂(x⊗x) + ... + Fd(x...⊗x)
%
%   Occasionally, we may wish to evaluate this and print the symbolic
%   expression f(x) to visualize it; this function does that. However,
%   small values will be printed by default using Matlab's vpa() command,
%   so here we round any values below a certain threshold down to make the
%   print command less cluttered.
%
%   TODO: Add support for sparseIJV, factoredMatrix, etc. coefficients
arguments 
    f
    nvp.n = size(f{1},2)
    nvp.variable = "x"
    nvp.thresh = 1e-7
    nvp.degree = length(f)
end
x = sym(nvp.variable,[nvp.n,1]);

% Sometimes the first or second coefficient may not be a standard class, so
% convert them to double/full.
if ~isa(f{1},'double')
    f{1} = double(f{1});
end
if ~isa(f{2},'double')
    f{2} = double(f{2});
end

% Round everything below thresh to zero to unclutter the print statement
for i=1:nvp.degree
    if isa(f{i},'double') % sparseIJV class gives issues, for now just do the doubles
        f{i}(abs(f{i}) < nvp.thresh) = 0;
    end
end
disp(vpa(kronPolyEval(f, x, nvp.degree), 3))
end