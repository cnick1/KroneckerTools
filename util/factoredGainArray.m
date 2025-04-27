classdef factoredGainArray < matlab.mixin.indexing.RedefinesBrace
    %FACTOREDGAINARRAY Array of polynomial coefficients with reduction basis
    %factor.
    %   This class is designed to aid in computing, storing, and evaluating
    %   efficiently the feedback law in PPR when reduction is used. The
    %   feedback law is written in an approximate form in terms of the
    %   reduced-order state x≈Txᵣ as
    %       u(x) = K₁x + K₂ᵣ(xᵣ⊗xᵣ) + K₃ᵣ(xᵣ⊗xᵣ⊗xᵣ) + ... + Kᵣᵈ⁻¹(...⊗xᵣ) )
    %   The full-order gain coefficients are therefore approximated as
    %       K₁ = K₁
    %       K₂ ≈ K₂ᵣ(T⁻¹⊗T⁻¹)             (Note: in practice, the T we are
    %       K₃ ≈ K₃ᵣ(T⁻¹⊗T⁻¹⊗T⁻¹)         using is orthogonal, so T⁻¹=Tᵀ)
    %   This class stores the gains {K₁,K₂ᵣ,...} and the transformation
    %   coefficient (and its inverse) {T,T⁻¹}.
    %
    %   An important feature of this class is its integration with the
    %   kronPolyEval() function. To evaluate u(x), kronPolyEval()
    %   identifies if K is a factoredGainArray, in which case it
    %   efficiently evaluates u(x) by handling the reduced-order state
    %   approximation for the higher-order coefficients. For example,
    %   K₃ ≈ K₃ᵣ(T⁻¹⊗T⁻¹⊗T⁻¹) is never formed; we instead form xᵣ=T⁻¹x and
    %   evaluate the higher-order terms directly in terms of xᵣ, which is
    %   much more efficient. This class allows the user to not have to
    %   worry about these details.
    %
    %   For convenience, this class also overloads the following methods so
    %   that a factoredGainArray behaves similarly to a regular cell array:
    %       - brace indexing: permits indexing as if the array were a
    %               regular cell array (currently only for single indexing)
    %       - disp: provides option for displaying the reduced OR the
    %               approximation of the full coefficients
    
    properties
        ReducedGains
        Tinv
    end
    
    methods (Access=protected)
        % Overloading {} indexing
        function varargout = braceReference(obj,indexOp)
            % [varargout{1:nargout}] = obj.ReducedGains.(indexOp); % Reduced gain
            k = indexOp.Indices{1}; % Hardcoded only for single indexing
            if k == 1
                [varargout{1:nargout}] = obj.ReducedGains.(indexOp);
            else
                [varargout{1:nargout}] = calTTv({obj.Tinv}, k, k, obj.ReducedGains{k}.').'; % Full gain
            end
        end
        
        function obj = braceAssign(obj,indexOp,varargin)
            if isscalar(indexOp)
                [obj.ReducedGains.(indexOp)] = varargin{:};
                return;
            end
            [obj.ReducedGains.(indexOp)] = varargin{:};
        end
        
        function n = braceListLength(obj,indexOp,indexContext)
            n = listLength(obj.ReducedGains,indexOp,indexContext);
        end
    end
    methods (Access=public)
        function obj = factoredGainArray(ReducedGains,Tinv)
            %FACTOREDGAINARRAY Construct an instance of this class
            %   A factoredGainArray is just a set of reduced gains and the
            %   transformation that converts from the full state to the
            %   reduced state. Currently, it is just a simple struct with
            %   two components:
            %       ReducedGains - cell array of the reduced gains
            %       Tinv         - (r x n) matrix used to construct the
            %                      reduced state xᵣ=T⁻¹x
            obj.ReducedGains = ReducedGains;
            obj.Tinv = Tinv;
        end
        
        function disp(obj, dispFull)
            %DISP Overloaded method to display a factoredGainArray
            % Optional argument:
            %   dispFull - boolean, defaults to false. If true, then
            %              Kᵢ ≈ Kᵢᵣ(T⁻¹⊗...⊗T⁻¹) will all be formed
            %              explicitly and displayed. Note: this is just for
            %              debugging, it should never typically be done.
            if nargin < 2
                dispFull = false;
            end
            
            if dispFull
                fullGains = cell(1,length(obj.ReducedGains));
                fullGains{1} = obj.ReducedGains{1};
                for k = 2:length(obj.ReducedGains)
                    fullGains{k} = calTTv({obj.Tinv}, k, k, obj.ReducedGains{k}.').';
                end
                disp(fullGains)
            else
                builtin('disp', obj);
                fprintf("    Call disp(K,true) to display the full coefficient array.\n\n")
            end
        end
        
        function result = length(obj)
            result = length(obj.ReducedGains);
        end
    end
end

