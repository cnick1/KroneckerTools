function s = dispPolyDynamics(f,g,h,nvp)
%dispPolyDynamics Display control-affine polynomial dynamics
%
%   Usage:     dispPolyDynamics(f,g,h)
%
%   Input Variables:
%          f,g,h - cell array containing drift, input, and output coefficients
%
%   Additional name/value pair options:
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
    g
    h
    nvp.thresh = 1e-7
    nvp.degree = length(f)
end
[p,n] = size(h{1});

fs = dispKronPoly(f,thresh=nvp.thresh,degree=nvp.degree);
if length(g) > 1
    gs = dispKronPoly(g(2:end),thresh=nvp.thresh,degree=nvp.degree);
else 
    gs = num2cell(zeros(1,n));
end
hs = dispKronPoly(h,thresh=nvp.thresh,degree=nvp.degree);

for i=1:n
    fprintf('      dx%i/dt = %s + (%s + %s) u\n',i,fs{i},num2str(g{1}(i)),gs{i})
end

for i=1:p
    fprintf('          y%i = %s\n',i,hs{i})
end


end