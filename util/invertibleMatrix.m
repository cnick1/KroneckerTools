classdef invertibleMatrix
    %INVERTIBLEMATRIX An n x n matrix M whose inverse M⁻¹ is known analytically
    %
    %   This class is designed to facilitate handling the matrix M and
    %   operations such as inv() and M\ directly using the known inverse M⁻¹.
    %   The goal is to internally just store M and Minv, but externally have
    %   something that looks and quacks like the matrix M. This is primarily
    %   done by overloading key methods so that the user doesn't have to worry
    %   about these details. We are simply going to overload the mldivide(),
    %   mrdivide(), and inv() functions to directly use the known inverse.
    %
    %   This class overloads the following methods so that an invertibleMatrix
    %   behaves similarly to a regular matrix:
    %       - invertibleMatrix: constructor, just pass in M and Minv
    %       - disp: display the matrix and its inverse
    %       - inv: return M⁻¹
    %       - mldivide: compute using matrix multiplication with Minv from left
    %       - mrdivide: compute using matrix multiplication with Minv from right
    %       - vec: vectorize M
    %       - mtimes: standard matrix multiplication
    %       - plus: standard matrix addition
    %
    %  See also: factoredMatrix, factoredMatrixInverse, factoredGainArray, factoredValueArray
    %%
    properties
        M
        Minv
    end

    methods

        % Constructor
        function obj = invertibleMatrix(M, Minv)
            obj.M = M;
            obj.Minv = Minv;
        end

        % Display M and Minv
        function disp(obj, dispFull)
            if nargin < 2
                dispFull = false;
            end

            if dispFull
                fprintf('  <a href="matlab:open invertibleMatrix">invertibleMatrix</a> with properties:\n')
                fprintf('  %s = \n', inputname(1))
                builtin('disp', obj.M);
                fprintf('  %s⁻¹ = \n', inputname(1))
                builtin('disp', obj.Minv);
            else
                builtin('disp', obj);
                fprintf("    Call disp(%s,true) to display the full matrix.\n\n", inputname(1))
            end
        end

        % Main overload #1: return inverse directly
        function result = inv(obj)
            result = obj.Minv;
        end

        % Vectorize
        function result = vec(obj)
            result = obj.M(:);
        end

        % Standard matrix multiplication
        function result = mtimes(obj, other)
            if isa(obj, 'invertibleMatrix') && isa(other, 'invertibleMatrix')
                % Both are invertibleMatrix; not planning on using this
                result = obj.M * other.M;
            elseif isa(other, 'invertibleMatrix')
                result = obj * other.M;
            else
                result = obj.M * other;
            end
        end

        % Left inverse
        function result = mldivide(obj, other)
            if isa(obj, 'invertibleMatrix')
                result = obj.Minv * other;
            elseif isa(other, 'invertibleMatrix')
                result = obj \ other.M;
            else % shouldn't ever be called
                result =  builtin('mldivide', obj, other);
            end
        end

        % Right inverse
        function result = mrdivide(obj, other)
            if isa(other, 'invertibleMatrix')
                result = obj * other.Minv;
            elseif isa(obj, 'invertibleMatrix')
                result = obj.M / other;
            else % shouldn't ever be called
                result =  builtin('mrdivide', obj, other);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% Standard algebraic operations
        function result = plus(obj, other)
            if isa(other, 'invertibleMatrix')
                other = other.M;
            end
            if isa(obj, 'invertibleMatrix')
                obj = obj.M;
            end
            result =  builtin('plus', obj, other);
        end

        function result = minus(obj, other)
            if isa(other, 'invertibleMatrix')
                other = other.M;
            end
            if isa(obj, 'invertibleMatrix')
                obj = obj.M;
            end
            result =  builtin('minus', obj, other);
        end

        %% Other 
        function result = det(obj)
            result = det(obj.M);
        end

        function result = isemtpy(obj)
            result = isemtpy(obj.M);
        end

        function result = length(obj)
            result = length(obj.M);
        end

        function result = numel(obj)
            result = numel(obj.M);
        end

        function result = size(obj)
            result = size(obj.M);
        end

        function result = double(obj)
            result = double(obj.M);
        end

    end

end
