function dispPolyDynamics(f,g,h,nvp)
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
    f cell
    g cell
    h cell
    nvp.thresh = 1e-7
    nvp.degree = length(f)
    nvp.variable = "x"
end
[p,n] = size(h{1});
[~,m] = size(g{1});

% Print the state equation
fs = dispKronPoly(f,thresh=nvp.thresh,degree=nvp.degree,variable=nvp.variable);

inputScenario = 'linear';
for i=2:length(g)
    if nnz(g{i}) > 0
        inputScenario = 'nonlinear';
    end
end
switch inputScenario
    case 'linear'
        for i=1:n
            fprintf('      d%s%i/dt = %s',nvp.variable,i,regexprep(fs{i},' +',' '))
            for j=1:m
                if abs(g{1}(i,j)) > 1e-7
                    fprintf(' + %s*u%i',num2str(g{1}(i,j)),j)
                end
            end
            fprintf('\n')
        end
    case 'nonlinear'
        % Convert g(x) to ∑ gᵢ(x)
        g_j = cell(m,length(g));
        for j=1:m
            for k=1:(length(g)-1)
                g_j{j,k+1} = g{k+1}(:,j:m:end);
            end
        end

        for i=1:n
            % Drift
            fprintf('      d%s%i/dt = %s',nvp.variable,i,regexprep(fs{i},' +',' '))
            for j=1:m
                fprintf(' + (')
                % Linear inputs
                if abs(g{1}(i,j)) > 1e-7
                    fprintf('%s',num2str(g{1}(i,j)))
                end
                % Nonlinear inputs
                gs = dispKronPoly(g_j(j,2:end),thresh=nvp.thresh,degree=nvp.degree,variable=nvp.variable);
                fprintf(' + %s ',regexprep(gs{i},' +',' '))
                fprintf(')u%i',j)
            end
            fprintf('\n')
        end

        % for i=1:n
        %     % Drift
        %     fprintf('      d%s%i/dt = %s',nvp.variable,i,fs{i})
        %     % Linear inputs
        %     for j=1:m
        %         if abs(g{1}(i,j)) > 1e-7
        %             fprintf(' + %s*u%i',num2str(g{1}(i,j)),j)
        %         end
        %     end
        %     % Nonlinear inputs
        %     for j=1:m
        %         gs = dispKronPoly(g_j(j,2:end),thresh=nvp.thresh,degree=nvp.degree,variable=nvp.variable);
        %         fprintf(' + %s u%i\n',gs{i},j)
        %     end
        %     fprintf('\n')
        % end
end



% Print the output equation
hs = dispKronPoly(h,thresh=nvp.thresh,degree=nvp.degree,variable=nvp.variable);
for i=1:p
    fprintf('          y%i = %s\n',i,regexprep(hs{i},' +',' '))
end


end