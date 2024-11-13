classdef factoredArray < matlab.mixin.indexing.RedefinesBrace
    %FACTOREDARRAY Array of polynomial coefficients with reduction basis
    %factor.
    %   Detailed explanation goes here

    properties
        ReducedGains
        Tinv
    end

    methods (Access=protected)
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
        function obj = factoredArray(ReducedGains,Tinv)
            %FACTOREDARRAY Construct an instance of this class
            %   Detailed explanation goes here
            obj.ReducedGains = ReducedGains;
            obj.Tinv = Tinv;
        end

        function disp(obj, dispFull)
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
                fprintf("    Call disp(Gains,true) to display the full coefficient array.\n\n")
            end
        end

        function result = length(obj)
            result = length(obj.ReducedGains);    
        end
    end
end

