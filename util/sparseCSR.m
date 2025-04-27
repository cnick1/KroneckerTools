classdef sparseCSR
    %sparseCSR Wrapper for storing short/wide sparse arrays by transposing them
    %   Matlab stores sparse arrays in Compressed Sparse Column (CSC)
    %   format. This means that it basically stores an m x n matrix as n
    %   sparse column vectors. This means that the storage requirements are
    %   drastically different for short/wide vs tall/skinny arrays. This
    %   can be verified easily:
    %   >> A=sparse(100,100^3);B=sparse(100^3,100); whos
    %        Name            Size             Bytes     Class     Attributes
    %         A          100x1000000         8000024    double    sparse
    %         B          1000000x100           824      double    sparse
    %   So even though these are both all zero sparse arrays, they have
    %   drastically different storage requirements. In fact, the number of
    %   rows has no impact on the storage, it is primarily the number of
    %   columns.
    %
    %   When writing Kronecker polynomials, we quickly encounter these
    %   short/wide arrays; it would be nice to have them stored in
    %   Compressed Sparse Row (CSR) format, but Matlab does not support
    %   this natively. In this class, we seek to provide a work-around by
    %   internally storing the transpose of a short/wide array, but then
    %   externally making it look like the short/wide array that we want.
    %   This is primarily done by overloading key methods to transpose the
    %   stored array whenever it is accessed.
    %
    %   Here is a list of the overloaded methods:
    %       constructor: S = sparseCSR(m,n) creates an all zero m x n
    %                        sparse array, the same as sparse(m,n). However,
    %                        it stores sparse(n,m) instead.
    %                    S = sparseCSR(i,j,s,m,n) is the same as
    %                        sparse(i,j,s,m,n), but stores the transpose of
    %                        the array.
    %                    For all other cases, we fall back on calling
    %                    sparse() and THEN transposing the result.
    %
    properties (Access = private)
        StoredTransposed
    end
    
    methods
        function obj = sparseCSR(varargin)
            % Custom sparse constructor to flip size before building
            
            if nargin == 2 && isnumeric(varargin{1}) && isnumeric(varargin{2})
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
                % Other calling forms: fallback (maybe sparse(i,j,s))
                A = sparse(varargin{:});
                obj.StoredTransposed = A.';  % Worst case: safe fallback
            end
        end
        
        function s = size(obj, dim)
            sz = size(obj.StoredTransposed);
            sz = fliplr(sz);
            if nargin == 2
                s = sz(dim);
            else
                s = sz;
            end
        end
        
        function result = length(obj)
            result = length(obj.StoredTransposed.');
        end
        
        function B = full(obj)
            B = full(obj.StoredTransposed.');
        end
        
        function varargout = find(obj, varargin)
            [varargout{1:nargout}] = find(obj.StoredTransposed.', varargin{:});
        end
        
        function spy(obj)
            spy(obj.StoredTransposed.');
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

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function varargout = subsref(obj, S)
            switch S(1).type
                case '()'
                    [i,j] = deal(S(1).subs{:});
                    varargout{1} = obj.StoredTransposed(j,i);
                    if length(S) > 1
                        [varargout{1:nargout}] = builtin('subsref', varargout{1}, S(2:end));
                    end
                case '.'
                    [varargout{1:nargout}] = builtin('subsref', obj, S);
                otherwise
                    error('Unsupported indexing type.')
            end
        end
        
        
    end
end
