classdef factoredMatrix
    properties
        Kr
        T
    end
    
    methods
        function obj = factoredMatrix(Kr, T)
            obj.Kr = Kr;
            obj.T = T;
        end
        
        function result = mtimes(obj, other)
            if isa(other, 'CustomMatrix')
                result = obj.Kr * obj.T * other.Kr * other.T;
            else
                result = obj.Kr * obj.T * other;
            end
        end

        function disp(obj, dispFull)
            if nargin < 2 
                dispFull = false;
            end

            if dispFull
                disp(obj.Kr * obj.T)
            else
                builtin('disp', obj);
                fprintf("    Call disp(K,true) to display the full K matrix.\n\n")
            end
        end

        % function disp(obj)
        %     disp(obj.Kr * obj.T);
        % end
    end
end
