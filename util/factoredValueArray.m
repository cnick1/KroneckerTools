classdef factoredValueArray < matlab.mixin.indexing.RedefinesParen & ...
        matlab.mixin.indexing.RedefinesBrace
    %FACTOREDVALUEARRAY Array of polynomial coefficients with reduction basis
    %factor.
    %   This class is designed to aid in computing, storing, and evaluating
    %   efficiently the value function in PPR when reduction is used. The
    %   value function is written in an approximate form in terms of the
    %   reduced-order state x≈Txᵣ as
    %       V(x) =    1/2 ( v₂ᵀ(x⊗x) + v₃ᵣᵀ(xᵣ⊗xᵣ⊗xᵣ) + ... +   vᵣᵈᵀ(...⊗xᵣ) )
    %   The full-order value coefficients are therefore approximated as
    %       v₂ = v₂                           (Note: in practice, the T we are
    %       v₃ ≈ (T⁻¹⊗T⁻¹⊗T⁻¹)ᵀ v₃ᵣ           using is orthogonal, so T⁻¹=Tᵀ)
    %   This class stores the gains {K₁,K₂ᵣ,...} and the transformation
    %   coefficient (and its inverse) {T,T⁻¹}.
    %
    %   An important feature of this class is its integration with the
    %   kronPolyEval() function. To evaluate V(x), kronPolyEval()
    %   identifies if K is a factoredValueArray, in which case it
    %   efficiently evaluates V(x) by handling the reduced-order state
    %   approximation for the higher-order coefficients. For example,
    %   v₃ ≈ v₃ᵣ(T⁻¹⊗T⁻¹⊗T⁻¹) is never formed; we instead form xᵣ=T⁻¹x and
    %   evaluate the higher-order terms directly in terms of xᵣ, which is
    %   much more efficient. This class allows the user to not have to
    %   worry about these details.
    %
    %   For convenience, this class also overloads the following methods so
    %   that a factoredValueArray behaves similarly to a regular cell array:
    %       - brace indexing: permits indexing as if the array were a
    %               regular cell array (currently only for single indexing)
    %       - parenthesis indexing: permits indexing as if the array were a
    %               regular cell array (currently only for single indexing)
    %       - disp: provides option for displaying the reduced OR the
    %               approximation of the full coefficients
    properties
        ReducedValueCoefficients
        Tinv
    end
    
    methods (Access=protected)
        function varargout = braceReference(obj,indexOp)
            k = indexOp.Indices{1}; % Hardcoded only for single indexing
            if k == 1 || k == 2
                [varargout{1:nargout}] = obj.ReducedValueCoefficients.(indexOp);
            else
                [varargout{1:nargout}] = calTTv({obj.Tinv}, k, k, obj.ReducedValueCoefficients{k}); % Full value fun
            end
        end
        
        function obj = braceAssign(obj,indexOp,varargin)
            if isscalar(indexOp)
                [obj.ReducedValueCoefficients.(indexOp)] = varargin{:};
                return;
            end
            [obj.ReducedValueCoefficients.(indexOp)] = varargin{:};
        end
        
        function n = braceListLength(obj,indexOp,indexContext)
            n = listLength(obj.ReducedValueCoefficients,indexOp,indexContext);
        end
        
        % Overloading () indexing
        function out = parenReference(obj, indexOp)
            out = obj;
            out.ReducedValueCoefficients = obj.ReducedValueCoefficients.(indexOp);
        end
        
        function obj = parenAssign(obj, indexOp, varargin)
            indices = indexOp.Indices{1};
            if isnumeric(indices) && numel(indices) == numel(varargin)
                for i = 1:numel(indices)
                    obj.ReducedValueCoefficients{indices(i)} = varargin{i};
                end
            else
                error("Indexing must match the number of input values.")
            end
        end
        
        function n = parenListLength(~,~,~)
            n = 1;
        end
        
        function obj = parenDelete(obj,indexOp)
            obj.ReducedValueCoefficients.(indexOp) = [];
        end
        
    end
    methods (Access=public)
        function obj = factoredValueArray(ReducedValueCoefficients,Tinv)
            %FACTOREDVALUEARRAY Construct an instance of this class
            %   A factoredValueArray is just a set of reduced value
            %   function coefficients and the transformation that converts
            %   from the full state to the reduced state. Currently, it is
            %   just a simple struct with two components:
            %       ReducedValueCoefficients - cell array of the reduced
            %                                  value function coefficients
            %       Tinv         - (r x n) matrix used to construct the
            %                      reduced state xᵣ=T⁻¹x
            obj.ReducedValueCoefficients = ReducedValueCoefficients;
            obj.Tinv = Tinv;
        end
        
        function out = cat(~,varargin)
            out = factoredValueArray;
            for ix = 1:length(varargin)
                tmp = varargin{ix};
                if isa(tmp,'factoredValueArray')
                    out.ReducedValueCoefficients = [out.ReducedValueCoefficients,tmp.ReducedValueCoefficients];
                else
                    out.ReducedValueCoefficients{end+1} = tmp;
                end
            end
        end
        
        function varargout = size(obj,varargin)
            [varargout{1:nargout}] = size(obj.ReducedValueCoefficients,varargin{:});
        end
        
        function disp(obj, dispFull)
            %DISP Overloaded method to display a factoredValueArray
            % Optional argument:
            %   dispFull - boolean, defaults to false. If true, then
            %              vᵢ ≈ (T⁻¹⊗...⊗T⁻¹)ᵀ vᵢᵣ will all be formed
            %              explicitly and displayed. Note: this is just for
            %              debugging, it should never typically be done.
            if nargin < 2
                dispFull = false;
            end
            
            if dispFull
                fullCoeffs = cell(1,length(obj.ReducedValueCoefficients));
                fullCoeffs{1} = obj.ReducedValueCoefficients{1};
                for k = 2:length(obj.ReducedValueCoefficients)
                    fullCoeffs{k} = calTTv({obj.Tinv}, k, k, obj.ReducedValueCoefficients{k});
                end
                disp(fullCoeffs)
            else
                builtin('disp', obj);
                fprintf("    Call disp(v,true) to display the full coefficient array.\n\n")
            end
        end
        
        function result = length(obj)
            result = length(obj.ReducedValueCoefficients);
        end
    end
end

