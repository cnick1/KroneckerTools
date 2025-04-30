classdef sparseCSR
    %sparseCSR Wrapper for storing short/wide sparse arrays by transposing them
    %
    %   Usage: Identical to sparse() (mostly)
    %
    %   Matlab stores sparse arrays in Compressed Sparse Column (CSC)
    %   format. This means that it basically stores an m x n matrix as n
    %   sparse column vectors. This means that the storage requirements are
    %   drastically different for short/wide vs tall/skinny arrays. This
    %   can be verified easily:
    %   >> A=sparse(100,100^3);B=sparse(100^3,100); whos
    %        Name           Size            Bytes     Class     Attributes
    %         A         100x1000000        8000024    double      sparse
    %         B         1000000x100          824      double      sparse
    %   So even though these are both all zero sparse arrays, they have
    %   drastically different storage requirements. In fact, the number of
    %   rows has no impact on the storage, it is primarily the number of
    %   columns.
    %
    %   When writing Kronecker polynomials, we quickly encounter these
    %   short/wide arrays; it would be nice to have them stored in
    %   Compressed Sparse Row (CSR) format, but Matlab does not support
    %   this natively. The goal is to internally just store transposed
    %   tall/skinny array, but externally have something that looks and
    %   quacks like a short/wide array. This is primarily done by
    %   overloading key methods.
    %
    %   Here is a list of the overloaded methods:
    %       Constructor:
    %           S = sparseCSR(m,n) creates an all zero m x n sparse array,
    %               the same as sparse(m,n).
    %           S = sparseCSR(i,j,s,m,n) is the same as sparse(i,j,s,m,n),
    %               but stores the transpose of the array.
    %           For all other cases, we fall back on calling
    %           sparse() and THEN transposing the result.
    %       Basic manipulation/indexing:
    %           size, length, find, norm, disp, transpose
    %       Sparse characteristics:
    %           full, spy, nnz, issparse
    %       Scalar algebra:
    %           plus, minus, times, rdivide
    %       Matrix algebra:
    %           plus, minus, mtimes, mldivide, mrdivide
    %
    %   Here is a list of the properties that this class can have:
    %           StoredTransposed - transposed array that is stored as a
    %                              standard sparse array
    %
    %   Part of the KroneckerTools repository.
    %%
    properties (Access = private)
        StoredTransposed
    end
    
    methods
        function obj = sparseCSR(varargin)
            % Custom sparse constructor to flip size before building
            
            if nargin == 1 && isstruct(varargin{1})
                obj.StoredTransposed = varargin{1}.StoredTransposed;
            elseif nargin == 2 && isnumeric(varargin{1}) && isnumeric(varargin{2})
                % Called like sparse(m,n)
                m = varargin{1};
                n = varargin{2};
                obj.StoredTransposed = sparse(n,m); % FLIP before calling sparse!
                
            elseif nargin >= 5 && isnumeric(varargin{1}) && isnumeric(varargin{2}) && isnumeric(varargin{3})
                % Called like sparse(i,j,s,m,n)
                i = varargin{1};
                j = varargin{2};
                s = varargin{3};
                m = varargin{4};
                n = varargin{5};
                % Flip i and j, and flip m and n
                obj.StoredTransposed = sparse(j,i,s,n,m);  % FLIP
            else
                % Other calling forms: fallback (maybe sparse(i,j,s) or sparse(S) for a matrix S)
                A = sparse(varargin{:});
                obj.StoredTransposed = A.';  % Worst case: safe fallback
            end
        end
        
        function varargout = size(obj, dim)
            sz = size(obj.StoredTransposed);
            sz = fliplr(sz);
            
            if nargin == 2
                varargout{1} = sz(dim);
            else
                if nargout <= 1
                    varargout{1} = sz;
                else
                    for k = 1:nargout
                        if k <= numel(sz)
                            varargout{k} = sz(k);
                        else
                            varargout{k} = 1; % MATLAB behavior: size returns 1 for extra dimensions
                        end
                    end
                end
            end
        end
        
        
        function result = length(obj)
            result = length(obj.StoredTransposed.');
        end
        
        function B = full(obj)
            B = full(obj.StoredTransposed.');
        end
        
        function varargout = find(obj, varargin)
            
            switch nargout
                case {0, 1}
                    [varargout{1:nargout}] = find(obj.StoredTransposed.', varargin{:}); % for linear indexing, we need to make sure we get the correct indexing by transposing first
                case {2, 3}
                    [varargout{1:nargout}] = find(obj.StoredTransposed, varargin{:});
                    varargout([1 2]) = varargout([2 1]);
            end
            
        end
        
        function result = norm(obj, norm_type)
            if nargin < 2
                norm_type = 'inf';
                fprintf('  (sparseCSR: defaulting to inf norm)\n')
            end
            
            switch norm_type
                case 'inf'
                    % Infinity norm: max row sum
                    result = norm(obj.StoredTransposed, 1); % inf norm = 1 norm for transpose (much faster)
                case 'fro'
                    % Frobenius norm: sqrt of sum of squares of all entries
                    result = norm(obj.StoredTransposed, 'fro');
                case 1
                    % 1-norm: max column sum
                    result = norm(obj.StoredTransposed, 'inf'); % 1 norm = inf norm for transpose (much faster)
                otherwise
                    warning("sparseCSR: potentially unsupported norm type.")
                    result = norm(obj.StoredTransposed.', norm_type);
            end
        end
        
        function spy(obj, varargin)
            if nargin > 1
                spy(obj.StoredTransposed.', varargin);
            else
                spy(obj.StoredTransposed.');
            end
        end
        
        function disp(obj)
            disp(obj.StoredTransposed.')
        end
        
        function n = nnz(obj)
            n = nnz(obj.StoredTransposed);
        end
        
        function n = issparse(~)
            n = true;
        end
        
        function result = transpose(obj)
            result = obj.StoredTransposed;
        end
        
        
        function varargout = subsref(obj, S)
            switch S(1).type
                case '()'
                    if isscalar(S(1).subs)
                        % Linear index: obj(i)
                        idx = S(1).subs{1};
                        if length(idx) > 1; error('sparseCSR: multiple linear indexing not currently supported'); end
                        [i,j] = ind2sub(size(obj),idx);
                        varargout{1} = obj.StoredTransposed(j,i);
                    elseif numel(S(1).subs) == 2
                        % 2D index: obj(i, j)
                        [i,j] = deal(S(1).subs{:});
                        varargout{1} = obj.StoredTransposed(j,i);
                    else
                        error('Unsupported indexing type.')
                    end
                    if length(S) > 1
                        [varargout{1:nargout}] = builtin('subsref', varargout{1}, S(2:end));
                    end
                    
                case '.'
                    [varargout{1:nargout}] = builtin('subsref', obj, S);
                    
                    
                case '{}'
                    error('Brace indexing is not supported for variables of this type.')
                    
                otherwise
                    error('Unsupported indexing type.')
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Standard algebraic operations
        function result = plus(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                result = plus(other, obj);
                return
            end
            if isscalar(other)
                result = obj.StoredTransposed.' + other; % will be converted to full
            elseif isa(other,'sparseCSR')
                result = sparseCSR((obj.StoredTransposed + other.StoredTransposed).');
            else
                result = sparseCSR(obj.StoredTransposed.' + other);
            end
        end
        
        function result = minus(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                % result = minus(other, obj)*-1;
                % return
                if isscalar(obj)
                    result = obj - other.StoredTransposed.'; % will be converted to full
                else
                    result.StoredTransposed = obj.' - other.StoredTransposed;
                    result = sparseCSR(result); % now just reclassify this struct as a sparseCSR object
                end
            else
                if isscalar(other)
                    result = obj.StoredTransposed.' - other; % will be converted to full
                elseif isa(other,'sparseCSR')
                    result = sparseCSR((obj.StoredTransposed - other.StoredTransposed).');
                else
                    % result = sparseCSR(obj.StoredTransposed.' - other);
                    result.StoredTransposed = obj.StoredTransposed - other.'; % do the operation on the transposed arrays
                    result = sparseCSR(result); % now just reclassify this struct as a sparseCSR object
                end
            end
        end
        
        function result = times(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                result = sparseCSR(obj .* other.StoredTransposed.');
            else
                result = sparseCSR(obj.StoredTransposed.' .* other);
            end
        end
        
        function result = rdivide(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                result = sparseCSR(obj ./ other.StoredTransposed.');
            else
                result = sparseCSR(obj.StoredTransposed.' ./ other);
            end
        end
        
        %% Standard linear algebra operations
        function result = mtimes(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                result = mtimes(other.', obj.').';
                return
            end
            % Currently just worried about handling case when
            % other is a vector; could make this more advanced
            if isscalar(other)
                result = sparseCSR(obj.StoredTransposed.' * other);
            else
                result = obj.StoredTransposed.' * other; % don't worry about casting it as sparseCSR because if it is a vector, second dimension will contract
            end
        end
        
        function result = mldivide(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                result = obj \ other.StoredTransposed.';
            else
                if isscalar(other)
                    result = sparseCSR(obj.StoredTransposed.' \ other);
                else
                    result = obj.StoredTransposed.' \ other;
                end
            end
        end
        
        function result = mrdivide(obj, other)
            if ~isa(obj,'sparseCSR') && isa(other,'sparseCSR')
                    result = obj / other.StoredTransposed.';
            else
                if isscalar(other)
                    result = sparseCSR(obj.StoredTransposed.' / other);
                else
                    result = obj.StoredTransposed.' / other;
                end
            end
        end
        
    end
end
