classdef sparseIJV
    %SPARSEIJV Class for storing sparse arrays using row/column indices and value arrays
    %
    %   Usage: Identical to sparse() (mostly)
    %
    %   Matlab stores sparse arrays in Compressed Sparse Column (CSC)
    %   format. This means that the storage requirements are drastically
    %   different for short/wide vs tall/skinny arrays. This can be
    %   verified easily:
    %     >> A=sparse(100,100^3);B=sparse(100^3,100);
    %     >> whos
    %         Name           Size            Bytes     Class     Attributes
    %          A         100x1000000        8000024    double      sparse
    %          B         1000000x100            824    double      sparse
    %   So even though these are both all zero sparse arrays, they have
    %   drastically different storage requirements. In fact, the number of
    %   rows has no impact on the storage, it is primarily the number of
    %   columns.
    %
    %   When writing Kronecker polynomials, we quickly encounter these
    %   short/wide arrays, so we need a way to handle them. For moderately
    %   large models, using Compressed Sparse Row (CSR) format (or a hack
    %   based on just transposing the array) works quite well; this is what
    %   the sparseCSR class does. However, to scale to extremely large
    %   models, even this is insufficient, as we hit the limit of the
    %   maximum allowable array size in Matlab. This is just because our
    %   huge sparse Kronecker coefficients, like F₃ ∈ (n x n³), are super
    %   wide; however, they are still sparse, so the number of nonzero
    %   entries is much smaller than n³.
    %
    %   This class is based on the observation that when we form F3 =
    %   sparseCSR(I, J, V, nvg, nvg^3), Matlab throws an error not when
    %   constructing the arrays I,J, and V (which contain all of the
    %   information needed), but rather when trying to form the entire
    %   array, just because the dimension is larger than the maximum
    %   permitted. So this class instead avoids using a Matlab sparse array
    %   altogether and just stores I,J, and V (hence sparseIJV) (we also
    %   need the dimensions). More precisely, this class is based on this
    %   from the sparse() documentation:
    %
    %        "This dissects and then reassembles a sparse matrix:
    %               [I,J,V] = find(S);
    %               [M,N] = size(S);
    %               S = sparse(I,J,V,M,N);"
    %
    %   The goal is to internally just store I,J,V,M, and N, but externally
    %   have something that looks and quacks like an array. This is primarily
    %   done by overloading key methods.
    %
    %   The result and benefit of this approach can be verified easily.
    %   Here are some examples from a custom nonlinear finite element code
    %   (getSystem29 in the PPR repository). We will consider the case of a
    %   2D mesh with 64 elements, leading to n=4225 degrees of freedom.
    %   First, (quite straightforward) for all zero arrays, e.g. the
    %   quadratic stiffness coefficient for this example, we see a major
    %   improvement (this is not unexpected; Matlab sparse arrays are not
    %   optimized to be empty after all):
    %     >> n=4225; S0=sparse(n,n^2); S1=sparseCSR(n,n^2); S2=sparseIJV(n,n^2);
    %     >> whos S0 S1 S2
    %         Name           Size            Bytes     Class     Attributes
    %          S0       4225x17850625     142805024    double      sparse
    %          S1       4225x17850625         33824   sparseCSR    sparse
    %          S2       4225x17850625            16   sparseIJV    sparse
    %   Just to reiterate, these are all zero arrays, i.e. the only things
    %   to store are the dimensions, but the way that the arrays are set up
    %   requires different amounts of memory. A more interesting case is an
    %   array that is not all zeros; the cubic stiffness coefficient in the
    %   finite element example is such a case. The I, J, and V values are
    %   computed in a loop that assembles the element matrices using the
    %   connectivity matrix. Then we can construct the full coefficient
    %   using the different sparse array classes:
    %     >> S1 = sparseCSR(I, J, V, n, n^3); S2 = sparseIJV(I, J, V, n, n^3);
    %     >> whos S1 S2
    %         Name          Size             Bytes     Class     Attributes
    %          S0           fails            > 4e12    double      sparse
    %          S1      4225x75418890625     4566048   sparseCSR    sparse
    %          S2      4225x75418890625    22020128   sparseIJV    sparse
    %   So actually sparseCSR is the best in this case memory wise; however,
    %   since it is still based on sparse(), it does hit a wall due to
    %   Matlab's maximum array size of 2^48 - 1. So for larger models, even
    %   though sparseIJV is less memory efficient than sparseCSR, it is the
    %   only option. (Either way, the efficient USE of these sparse arrays
    %   ---e.g. in PPR when computing the ROM dynamics or when simulating
    %   with kronPolyEval---is based on calling find() to get the nonzero
    %   entries, so it is possible that sparseIJV actually is more efficient
    %   because you don't then have to store the results when you call find,
    %   it should just be using pointers to the data fields in the struct.)
    %
    %   Here is a list of the overloaded methods:
    %       Constructor:
    %           S = sparseIJV(m,n) creates an all zero m x n sparse array,
    %               the same as sparse(m,n).
    %           S = sparseIJV(i,j,s,m,n) is the same as sparse(i,j,s,m,n),
    %               but stores the transpose of the array.
    %           S = sparseIJV(M) converts a matrix M to a sparseIJV matrix
    %           For all other cases, we fall back on calling sparseCSR().
    %           Note: M = sparse(S) converts a sparseIJV to a normal sparse
    %       Basic manipulation/indexing:
    %           size, length, find, norm, disp, transpose
    %       Sparse characteristics:
    %           sparse, full, spy, nnz, issparse
    %       Scalar algebra:
    %           plus, minus, times, rdivide
    %       Matrix algebra:
    %           plus, minus, mtimes, mldivide, mrdivide
    %
    %   Here is a list of the properties that this class can have:
    %           I - row indices
    %           J - column indices
    %           V - values
    %           M - total number of rows
    %           N - total number of columns
    %
    %   Part of the KroneckerTools repository.
    %%
    properties (Access = private)
        I
        J
        V
        M
        N
    end
    
    methods (Access = public)
        function obj = sparseIJV(varargin)
            if nargin == 1 && isstruct(varargin{1})
                % Externally constructed as a struct, just convert it to a sparseIJV class
                obj.I = varargin{1}.I;
                obj.J = varargin{1}.J;
                if isscalar(varargin{3})
                    obj.V = varargin{1}.V * ones(length(obj.I),1);
                else
                    obj.V = varargin{1}.V;
                end
                obj.M = varargin{1}.M;
                obj.N = varargin{1}.N;
            elseif nargin == 2 && isnumeric(varargin{1}) && isnumeric(varargin{2})
                % Called like sparse(m,n)
                obj.M = varargin{1};
                obj.N = varargin{2};
                obj.I = []; obj.J = []; obj.V = []; % all zeros sparse array is empty
            elseif nargin >= 5 && isnumeric(varargin{1}) && isnumeric(varargin{2}) && (isnumeric(varargin{3}) || islogical(varargin{3}))
                % Called like sparse(i,j,s,m,n)
                obj.I = varargin{1};
                obj.J = varargin{2};
                if isscalar(varargin{3})
                    obj.V = varargin{3} * ones(length(obj.I),1);
                else
                    obj.V = varargin{3};
                end
                obj.M = varargin{4};
                obj.N = varargin{5};
            elseif nargin == 1 && ismatrix(varargin{1})
                % Passing in a matrix and want to convert it to sparseIJV
                [obj.I,obj.J,obj.V] = find(varargin{1});
                [obj.M,obj.N] = size(varargin{1});
            else
                % Other calling forms: fallback to sparseCSR (maybe sparse(i,j,s))
                warning("SparseIJV: Unexpected constructor options, attempting to fallback to sparseCSR; here is a picture of my dog")
                c = ['@HE@JCME9KUHSDSAVCXN<OY;LV5?Q:Ql6e.`m.gw-Fo{2Hp{.GUw*6HX{-Eg{-Es+Gx,g|,lz<m|+D`{*>Qs*=Lo,:ES*8EOu+:JTt*8KUy+;Lm|,9Fm0?Ip4BLWz0=GQ[{2<FQ[w4>R\v3=R\v2?T^w4N[pBVh{4M\m7AYkx;M\o;EUgu6@JTft=GR\o}<FQ[q9CMWj~DP]gz@Lam<I`jw@Nn~DQp~EftKjw'
                    'A<FAKD:F:L8JTETBWDYXBPZ<MW6@R0;Rm7f/an/ix.Gq|3Kq|2HXx+7IY|2Fh|1Fx,gy-h}-m{*Bn},Ea|+?Xt+>Mr-;Fm+9FPv,;KUu+9Ljz,?Mn~-;Gn1@Jt5CMX{1>HRj|3=GR\z5?S]w4>S]w3@U_z=O\tCWi}5N]o8BZlz<N]p<HVhw7AKUgu>HSdq~=GR`s:DNXkGQ^i{AMbn=JakyAQoHdqHgu@Lkx'
                    '>=GBLE;G;M9KUJUC4E5YCQ3=NX7AS1?W:h0bo0jz/Hr}6Pr}3Ihy,8Ji}3Gk}2Gy-hz.iBn|+Cq~8Fk},EYu,?Ns.<Gn,:GTy-=LVv,BMk{-ANo.=Ho2AKv6DNj|2?ISk4>HS]{6@T^z5?T^z4EVi{CP]vDXk~6O^p9C[n{3=Tfs=IWiy8BLVhy?ITes>HSet;EOYm9HR_j~BNcr>KblzBbpIerIhvAMmy'
                    '?>HC;F<H<N:L7KVD5F:5DR4>OY8BT2@X;i1cp1n{1Tt~7Qs~4Jjz-?Kk~4Hl~3Qz.i}AlCq},Dr9Gn~-F\x-BOu/=Io-=HW{.BMky-CNl~.BQp/>Kp3BLw-7EOk3@JTm5?ITj|7AUi{6@Ui{8FWj|DT^wEYl7U_s:D\o|4>Ugt4>JXjz9CMWjz6@JUft?ITgx<FPZn:IS`kCOds?Lcm~CcqJfuJiwBNn{'
                    '@?ID<G=I@O;M8MWE6P;6ES5?PZ9CU3AY<m2dq5o|5fu8Tu5Kk{.@Ql5Kp4g|AlBmDr~-Et:Mr.K]|.CPx0>Ks.>Il~/CNl{.DOn/CRs0?Lt6CMz.9FQm4AKUo6@JUk8BVj|7AVj|9GXl~EU_z2MZm8Vgt;L]p}5?Vhu5?KYk|:DNXk{7AKVgu@JUhy=GQ[s;JTamFPew@MdnDdr>Kgv?KjxCOo|'
                    'A@JE=H>KAP@N9N5J7QC7FT6@Q[:DV4Cg/=n5e6p}6gv:Uv9Po|1CRp6Qq5h}BmCqEs7Fw;Nt9L`}/DQ~5?Lu/?Jm5DOm~/EPo0DSt1AMv8DN{/:GRo5BLVp7AKVm9CWl8BWl~:HYmFVi{3N[o@Whv<M^s~6@Wiw6@LZl;EOYl|8BLWhx7AKVjz>HRat<KUbn;GQfxANepIev?Lhw@Mk|DPp}'
                    'BA<F>IALBQAO:O6P8TD8KU7AR1;MW5Mh0>o6f8q8iwChw*CQp}2DSr7Rr6kCpDsFt8^x<Ou:Ma~6ER6AM~0@Kn6EPn0FQp1ETv2BNy,9Ej|0=HSp6CMWt8BLWo:DXm9CXm;IZoGWk|4O\pAXiw3=Ugt7AXjx7AM[n<FPZn}9CMXjy8BLWk{?ISew=LVcp<HRgzBOfqJfw@MixANn}Gfq~'
                    'DB=G?JBMCRBPAP7S@UE9LV8BS2<NX7Ni1?7g9r9jx,Djx+DRr~3ETs*@Ss*@lDqEthw9_y=]x;Nn:Fc7BN5BLo7FQo5GRs6FUy6Cjz-;Fk1>ITt7DNXv/9CMXp1;EYo:GYo<J[pHXl}5S]tBYj{4>Vhu8BYkz8BN\o=GQ[o~:DOYkz9CNXm~@JTfx>MXds=I^h~CPgr=KgyANjyBOo~Hgt'
                    'EC>H@KCNDSCQBQ8TAVL:MW9CT3=OY8Oj2c^heuCky-Fky,ESs4FUy+CTx+CpEs*Fw*ix:`zB^y<Oo;Gd*8CO6CMp8GRp6HSt7Gkz*7Dk{.=Gm2?JUv.:EOYw0:DOYt2<PZp1;HZp=O\t2IYm~6T^v2CZk|5?Wiv9CZl{9CQes>HRdq;EPZm{:DOYnAKUgz?NYew>J_iGQhs>LhzBOn|CPpIhu'
                    'GD?IBLDSITDRCR9UBWM;NX:DU4>PZ9Pk3d_i-fv,Dnz/Goz-FTv5GWz,DUz,Dq*Fw+P{+jy;j{*C_z=Pr<Kn+9DP7DNs*9HSs7ITv*8Kl{+8El|/>Ho3@KVw/<FPZz1;EPZv3=Q[t2<Q[t1>P]v3JZoAU_w3D[l~6@Xjw:J[n}:DTft5?ISes<FQ[n|;EPZo8BLVi{CO\fx?K`jHRiv?Ml}CPo}DeqJiv'];
                i = double(c(:)-32);
                j = cumsum(diff([0; i])<=0) + 1;
                spy(sparseIJV(i,j,1,100,67)); title("(my dog, not your sparseIJV array)")
                
                S = sparseCSR(varargin{:});
                [obj.I,obj.J,obj.V] = find(S);
                [obj.M,obj.N] = size(S);
            end
        end
        
        function varargout = subsref(obj, S)
            switch S(1).type
                case '()'
                    if isscalar(S(1).subs)
                        % Linear indexing: obj(i)
                        k = S(1).subs{1};
                        if length(k) > 1; error('sparseIJV: multiple linear indexing not currently supported'); end
                        [i, j] = ind2sub([obj.M, obj.N], k);
                        val = getValueAt(obj, i, j);
                        varargout{1} = val;
                        
                    elseif numel(S(1).subs) == 2
                        % 2D index: obj(i, j)
                        i = S(1).subs{1};
                        j = S(1).subs{2};
                        
                        % Handle ':' shorthand
                        if ischar(i) && strcmp(i, ':')
                            i = 1:obj.M;
                        end
                        if ischar(j) && strcmp(j, ':')
                            j = 1:obj.N;
                        end
                        
                        % Return a sparse matrix result
                        val = sparse(length(i), length(j));
                        [I_grid, J_grid] = ndgrid(i, j); % needed to get all pairs
                        linear_I = I_grid(:);
                        linear_J = J_grid(:);
                        
                        % Lookup each (i,j) pair
                        for idx = 1:numel(linear_I)
                            val(idx) = getValueAt(obj, linear_I(idx), linear_J(idx));
                        end
                        
                        varargout{1} = reshape(val, size(I_grid));
                    else
                        error('Only 1D or 2D indexing supported.')
                    end
                    
                    if length(S) > 1
                        [varargout{1:nargout}] = builtin('subsref', varargout{1}, S(2:end));
                    end
                    
                case '.'
                    [varargout{1:nargout}] = builtin('subsref', obj, S);
                    
                case '{}'
                    error('Brace indexing is not supported for variables of this type.');
                    
                otherwise
                    error('Unsupported indexing type.');
            end
        end
        
        function obj = subsasgn(obj, S, value)
            switch S(1).type
                case '()'
                    if numel(S(1).subs) ~= 2
                        warning('sparseIJV: only 2D indexing assignment is currently supported; converting to 2D indexing');
                        if iscell(S(1).subs) 
                            S(1).subs = S(1).subs{1};
                        end
                        [i, j] = ind2sub([obj.M, obj.N], S(1).subs);
                    else
                        i = S(1).subs{1};
                        j = S(1).subs{2};
                    end
                    
                    if ischar(i) && strcmp(i, ':'); i = 1:obj.M; end
                    if ischar(j) && strcmp(j, ':'); j = 1:obj.N; end
                    
                    [Igrid, Jgrid] = ndgrid(i, j);
                    linear_i = Igrid(:);
                    linear_j = Jgrid(:);
                    value = value(:);
                    
                    if isscalar(value)
                        % Expand scalar to full size
                        value = repmat(value, length(linear_i), 1);
                    elseif length(value) ~= length(linear_i)
                        error('sparseIJV: Value size does not match indexing.');
                    end
                    
                    % Remove existing entries at those (i,j)
                    mask = true(length(obj.I), 1);
                    for k = 1:length(obj.I)
                        if any(obj.I(k) == linear_i & obj.J(k) == linear_j)
                            mask(k) = false;
                        end
                    end
                    
                    obj.I = obj.I(mask);
                    obj.J = obj.J(mask);
                    obj.V = obj.V(mask);
                    
                    % Add new values, excluding zeros
                    nonzero = value ~= 0;
                    obj.I = [obj.I; linear_i(nonzero)];
                    obj.J = [obj.J; linear_j(nonzero)];
                    obj.V = [obj.V; value(nonzero)];
                    
                otherwise
                    obj = builtin('subsasgn', obj, S, value);
            end
        end
        
        function varargout = size(obj, dim)
            sz = [obj.M, obj.N];
            
            if nargin == 2
                varargout{1} = sz(dim);
            else
                if nargout <= 1
                    varargout{1} = sz;
                else
                    varargout = cell(nargout,1);
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
            result = max([obj.M, obj.N]);
        end
        
        function B = full(obj)
            warning("sparseIJV: converting large sparseIJV to full is probably a bad idea")
            if obj.N > obj.M
                B = full(sparseCSR(obj.I, obj.J, obj.V, obj.M, obj.N));
            else
                B = full(sparse(obj));
            end
        end
        
        function S = sparse(obj)
            S = sparse(obj.I, obj.J, obj.V, obj.M, obj.N);
        end
        
        function varargout = find(obj, varargin)
            switch nargout
                case {0, 1}
                    % Convert to linear indexing
                    [varargout{1:nargout}] = sub2ind([obj.M, obj.N], obj.I, obj.J);
                case 2
                    % Return [I, J]
                    varargout{1} = obj.I;
                    varargout{2} = obj.J;
                case 3
                    % Return [I, J, V]
                    varargout{1} = obj.I;
                    varargout{2} = obj.J;
                    varargout{3} = obj.V;
                otherwise
                    error('Too many output arguments.');
            end
        end

        function result = abs(obj)
            result = sparseIJV(obj.J, obj.I, abs(obj.V), obj.N, obj.M);
        end

        % function result = gt(obj, value) % S > value
        %     result = sparseIJV(obj.J, obj.I, gt(obj.V, value), obj.N, obj.M);
        % end
        % 
        % function result = lt(obj, value) % S < value
        %     result = sparseIJV(obj.J, obj.I, lt(obj.V, value), obj.N, obj.M);
        % end
        % 
        % function result = ge(obj, value) % S >= value
        %     result = sparseIJV(obj.J, obj.I, ge(obj.V, value), obj.N, obj.M);
        % end
        % function result = le(obj, value) % S <= value
        %     result = sparseIJV(obj.J, obj.I, le(obj.V, value), obj.N, obj.M);
        % end
        
        function result = norm(obj, norm_type)
            if nargin < 2
                norm_type = 'inf';
                fprintf('  (sparseIJV: defaulting to inf norm)\n')
            end
            
            switch norm_type
                case 'inf'
                    % Infinity norm: max row sum
                    row_sums = accumarray(obj.I, abs(obj.V), [obj.M, 1]);
                    result = max(row_sums);
                case 'fro'
                    % Frobenius norm: sqrt of sum of squares of all entries
                    result = norm(obj.V, 2);  % Equivalent to sqrt(sum(abs(V).^2))
                case 1
                    % 1-norm: max column sum
                    col_sums = accumarray(obj.J, abs(obj.V), [obj.N, 1]);
                    result = max(col_sums);
                otherwise
                    error("sparseIJV: only '1', 'inf', and 'fro' norms are supported.")
            end
        end
        
        function spy(obj)
            figure; % modified custom simplified version of Matlab's spy for sparseIJV
            if ~isempty(obj.I)
                maxSize = max(obj.M+1, obj.N+1);
                markersize = round(8*(log10(2500./maxSize)));
                markersize = max(4, min(14, markersize));
                p = plot(obj.J,obj.I,'marker','.','markersize',markersize,'linestyle','none','SeriesIndex',1);
                delete(datatip(p)); % Start datatip mode
                p.DataTipTemplate.DataTipRows = p.DataTipTemplate.DataTipRows([2 1]);
                p.DataTipTemplate.DataTipRows(1).Label = getString(message('MATLAB:spy:Row'));
                p.DataTipTemplate.DataTipRows(2).Label = getString(message('MATLAB:spy:Column'));
                p.DataTipTemplate.DataTipRows(3) = dataTipTextRow(getString(message('MATLAB:spy:Value')), string(obj.V));
            end
            xlabel(['nz = ' int2str(nnz(obj))]);
            % Aspect ratio is never more than a factor 10 between rows and columns
            aspectRatio = [min(obj.N+1, 10*(obj.M+1)), min(obj.M+1, 10*(obj.N+1)), 1];
            set(gca,'xlim',[0 obj.N+1],'ylim',[0 obj.M+1],'ydir','reverse','plotboxaspectratio',aspectRatio);
        end
        
        function disp(obj)
            if isempty(obj.I)
                fprintf("   All zero sparse: %i×%i\n\n",obj.M, obj.N)
            else
                % Vectorized approach
                fprintf("   (%i,%i)        %1.4f\n", [obj.I(:), obj.J(:), obj.V(:)].');  % Transpose to match fprintf column-wise reading
                fprintf("\n")
                
                % Loop-based approach
                % for i = 1:length(obj.V)
                %     fprintf("   (%i,%i)        %1.4f\n", obj.I(i), obj.J(i), obj.V(i))
                % end
                
            end
        end
        
        function n = nnz(obj)
            n = length(obj.I);
        end
        
        function n = issparse(~)
            n = true;
        end
        
        function result = transpose(obj)
            result = sparseIJV(obj.J, obj.I, obj.V, obj.N, obj.M);
        end
        
        function val = getValueAt(obj, i, j)
            % Helper for indexing
            idx = find(obj.I == i & obj.J == j, 1);
            if isempty(idx)
                val = 0;
            else
                val = obj.V(idx);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Standard algebraic operations
        function result = plus(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                result = plus(other, obj);
                return
            end
            if isa(other,'sparseIJV')
                if obj.M ~= other.M || obj.N ~= other.N
                    error('Matrix dimensions must agree.');
                end
                
                % Combine (I, J, V) entries from both objects
                I_combined = [obj.I; other.I];
                J_combined = [obj.J; other.J];
                V_combined = [obj.V; other.V];
                
                % Combine duplicates: create unique linear indices
                linInd = sub2ind([obj.M, obj.N], I_combined, J_combined);
                [uniqueInd, ~, idx] = unique(linInd);
                
                % Sum duplicate values
                V_summed = accumarray(idx, V_combined);
                
                % Recover row and column indices
                [I_final, J_final] = ind2sub([obj.M, obj.N], uniqueInd);
                
                % Construct new sparseIJV object
                result = sparseIJV(I_final, J_final, V_summed, obj.M, obj.N);
            else
                result = sparse(obj) + other; % will be converted to full
            end
        end
        
        function result = minus(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                result = obj - sparse(other);
                return
            end
            if isa(other,'sparseIJV')
                if obj.M ~= other.M || obj.N ~= other.N
                    error('Matrix dimensions must agree.');
                end
                
                % Combine (I, J, V) entries from both objects
                I_combined = [obj.I; other.I];
                J_combined = [obj.J; other.J];
                V_combined = [obj.V; -other.V];
                
                % Combine duplicates: create unique linear indices
                linInd = sub2ind([obj.M, obj.N], I_combined, J_combined);
                [uniqueInd, ~, idx] = unique(linInd);
                
                % Sum duplicate values
                V_summed = accumarray(idx, V_combined);
                
                % Recover row and column indices
                [I_final, J_final] = ind2sub([obj.M, obj.N], uniqueInd);
                
                % Construct new sparseIJV object
                result = sparseIJV(I_final, J_final, V_summed, obj.M, obj.N);
            else
                result = sparse(obj) - other; % will be converted to full
            end
        end
        
        function result = times(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                result = times(other,obj);
            elseif isa(obj,'sparseIJV') && isscalar(other)
                result = obj;
                result.V = result.V .* other;
            elseif isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                if obj.M ~= other.M || obj.N ~= other.N
                    error('Matrix dimensions must agree.');
                end
                
                % Find common nonzero (i,j) pairs
                lin1 = sub2ind([obj.M, obj.N], obj.I, obj.J);
                lin2 = sub2ind([other.M, other.N], other.I, other.J);
                [commonInd, ia, ib] = intersect(lin1, lin2);
                
                if isempty(commonInd)
                    % Return an empty sparseIJV
                    result = sparse([],[],[], obj.M, obj.N);
                    return
                end
                
                % Recover (i,j) and compute v = v1 .* v2
                i = obj.I(ia);
                j = obj.J(ia);
                v = obj.V(ia) .* other.V(ib);
                
                % Form result
                result = sparse(i, j, v, obj.M, obj.N);
            else
                result = sparse(obj) * other;
            end
        end
        
        function result = rdivide(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                result = obj ./ sparse(other) ;
            elseif isa(obj,'sparseIJV') && isscalar(other)
                result = obj;
                result.V = result.V ./ other;
            else
                result = sparse(obj) ./ other;
            end
        end
        
        %% Standard linear algebra operations
        function result = mtimes(A, B)
            if (isa(A,'sparseIJV') && ~nnz(A)) || (isa(B,'sparseIJV') && ~nnz(B))
                m = size(A, 1);
                n = size(B, 2);
                result = sparseIJV(m,n);
                return
            end
            if ~isa(A,'sparseIJV') && isa(B,'sparseIJV')
                % This is based on the columns of AB being linear combinations of the columns of A weighted by the entries in the columns of B
                % Sort B by J (column index) for fast grouping
                [sorted_J, sort_idx] = sort(B.J);
                sorted_I = B.I(sort_idx);
                sorted_V = B.V(sort_idx);
                
                % Find the split points between columns
                [unique_cols, first_idx, ~] = unique(sorted_J, 'first');
                last_idx = [first_idx(2:end)-1; numel(sorted_J)];
                
                % Preallocate lists for results
                res_I = cell(numel(unique_cols), 1);
                res_J = cell(numel(unique_cols), 1);
                res_V = cell(numel(unique_cols), 1);
                
                % Loop over each unique column of 'B'
                for k = 1:numel(unique_cols)
                    c = unique_cols(k);
                    idx_range = first_idx(k):last_idx(k);
                    
                    rows_in_col = sorted_I(idx_range);
                    vals_in_col = sorted_V(idx_range);
                    
                    % Compute the c-th column of the result
                    col_vec = A(:, rows_in_col) * vals_in_col;
                    
                    % Find nonzeros in the resulting column
                    nz_rows = find(col_vec);
                    nz_vals = col_vec(nz_rows);
                    
                    % Store the nonzero entries
                    res_I{k} = nz_rows(:);
                    res_J{k} = repmat(c, numel(nz_rows), 1);
                    res_V{k} = nz_vals(:);
                end
                
                % Create output sparseIJV object
                result = sparseIJV(vertcat(res_I{:}), vertcat(res_J{:}), vertcat(res_V{:}), size(A, 1), B.N);
                
            elseif isscalar(A) || isscalar(B)
                result = times(A, B);
            else % A*B where A is sparseIJV and B is something else
                % ABij = A(i,:)*B(:,j)
                
                % Get unique nonzero rows of A
                rows = unique(A.I).';
                
                % Preallocate lists to collect result entries
                max_nz_rows = length(rows);
                res_I = cell(max_nz_rows, 1);
                res_J = cell(max_nz_rows, 1);
                res_V = cell(max_nz_rows, 1);
                
                k = 0;
                for r = rows % Loop over each nonzero row
                    % Construct rth row of A as a Matlab sparse
                    % Find indices for row r in A
                    idx = find(A.I == r);
                    row_vec = sparse(1, A.J(idx), A.V(idx), 1, A.N);
                    
                    % Multiply with B (1 x N) * (N x P) = (1 x P)
                    prod_row = row_vec * B;
                    
                    % Find nonzeros in the resulting row
                    nz_cols = find(prod_row);
                    nz_vals = prod_row(nz_cols);
                    
                    % Append to result
                    k = k + 1;
                    res_I{k} = repmat(r, numel(nz_cols), 1);
                    res_J{k} = nz_cols(:);
                    res_V{k} = nz_vals(:);
                end
                
                % Create output sparseIJV object
                result = sparseIJV(vertcat(res_I{:}), vertcat(res_J{:}), vertcat(res_V{:}), A.M, size(B, 2));
            end
        end
        
        function result = mldivide(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                % A \ S, where A is dense and S is sparseIJV
                % Identify nonzero columns of S
                nonzero_cols = unique(other.J);
                num_cols = numel(nonzero_cols);
                num_rows = size(obj, 1);
                
                % Precompute row and column indices
                newI = repmat((1:num_rows)', num_cols, 1);
                newJ = repelem(nonzero_cols(:), num_rows, 1);
                
                % Accumulate values
                newV = cell(num_cols,1);
                
                for k = 1:num_cols
                    col = nonzero_cols(k);
                    mask = (other.J == col);
                    i_col = other.I(mask);
                    v_col = other.V(mask);
                    
                    b = zeros(num_rows, 1);
                    b(i_col) = v_col;
                    
                    newV{k} = obj \ b;
                end
                
                result = sparseIJV(newI, newJ, cell2mat(newV), num_rows, other.N);
            else
                result = sparse(obj) \ other; % shouldn't ever happen, but just for good measure fall back on sparse
            end
        end
        
        function result = mrdivide(obj, other)
            if ~isa(obj,'sparseIJV') && isa(other,'sparseIJV')
                result = obj / sparse(other); % shouldn't ever happen, but just for good measure fall back on sparse
            else
                if isscalar(other)
                    result = sparse(obj) ./ other;
                else
                    % S / A, where A is dense and S is sparseIJV
                    % Identify nonzero columns of S
                    nonzero_rows = unique(obj.I);
                    num_rows = numel(nonzero_rows);
                    num_cols = size(other, 2);
                    
                    % Precompute row and column indices
                    newI = repelem(nonzero_rows(:), num_cols, 1);
                    newJ = repmat((1:num_cols)', num_rows, 1);
                    
                    % Accumulate values
                    newV = cell(1,num_rows);
                    
                    for k = 1:num_rows
                        row = nonzero_rows(k);
                        mask = (obj.I == row);
                        j_row = obj.J(mask);
                        v_row = obj.V(mask);
                        
                        b = zeros(1, num_cols);
                        b(j_row) = v_row;
                        
                        newV{k} = b / other;
                    end
                    
                    result = sparseIJV(newI, newJ, cell2mat(newV).', obj.M, num_cols);
                end
            end
        end
        
    end
end
