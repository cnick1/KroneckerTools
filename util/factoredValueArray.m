classdef factoredValueArray < matlab.mixin.indexing.RedefinesParen & ...
        matlab.mixin.indexing.RedefinesBrace
    %FACTOREDVALUEARRAY Array of polynomial coefficients with reduction basis
    %factor.
    %   Detailed explanation goes here

    properties
        ReducedValueCoefficients
        Tinv
    end

    methods (Access=protected)
        function varargout = braceReference(obj,indexOp)
            % [varargout{1:nargout}] = obj.ReducedGains.(indexOp); % Reduced gain 
            k = indexOp.Indices{1}; % Hardcoded only for single indexing
            if k == 1 || k == 2
                [varargout{1:nargout}] = obj.ReducedValueCoefficients.(indexOp);
            else 
                [varargout{1:nargout}] = calTTv({obj.Tinv}, k, k, obj.ReducedValueCoefficients{k}); % Full gain
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
            %FACTOREDGAINARRAY Construct an instance of this class
            %   Detailed explanation goes here
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

