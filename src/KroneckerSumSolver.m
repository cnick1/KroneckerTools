function [x] = KroneckerSumSolver(A,b,k,E,M,solver)
%KroneckerSumSolver Efficiently solves a Kronecker sum system ℒₖ(A)x=b.
%
%   Usage: x = KroneckerSumSolver(A,b,k)
%
%   Inputs:
%       A      - cell array containing the coefficient matrix of the linear
%                system. Currently the implementation assumes each A is identical.
%                  • [Acell{1:k}] = deal(A) assigns A to each entry
%                    in the cell array without duplicating memory usage.
%       b      - RHS vector of the linear system
%       k      - desired degree of the computed value function.  The
%                default is length(A)
%       E      - mass matrix, for generalized Bartels-Stewart
%       M      - diagonal shift of the linear system.
%       solver - string specifying the solver. Options:
%                  • 'bartels-stewart' - Jeff's N-way Bartels-Stewart
%                     direct solver from [1]. Described more below.
%                  • 'chen-kressner' - calls laplace_merge() solver from
%                     [2]. This performs better and is the default.
%   Output:
%       x      - solution to x = ℒₖ(A)⁻¹b
%
%   Background: Define the k-way Lyapunov Matrix
%
%             |---------------------------------k terms-------------|
%   ℒₖ(A) = ( A⊗ ... ⊗Iₚ  +  Iₚ⊗A⊗ ... ⊗Iₚ  +  ...  +  Iₚ⊗ ... ⊗A )
%           |--k factors--|   |--k factors--|     ...   |--k factors--|
%
%   Then we have the linear system ℒₖ(A)x = b. Currently we assume the A
%   matrices are all the same, but they could be d different matrices of the
%   same size, so the function is written to take a cell array containing
%   the A matrices of size (n x n). The vectors b and x are size (n^d,1).
%
%   For now, we assume each A{i} is the same (an N-Way Lyapunov equation) but
%   this can be relaxed by performing a Schur factorization to each A{i}
%   and performing the proper change of variables.
%
%   Implements a k-Way version of the Bartels-Stewart algorithm
%   for a special Kronecker sum system (the special case all matrices are
%   the same) with an additional diagonal shift. M is an n x n matrix that is
%   added to block diagonal:  diag(M) = kron(eye(n^(d-1)), M)
%
%
%   Authors: Jeff Borggaard, Virginia Tech (Original author)
%            Nicholas Corbin, UC San Diego (Update to include E matrix, added Chen/Kressner solver functionality)
%
%
%`  References: [1] J. Borggaard and L. Zietsman, "On approximating
%               polynomial-quadratic regulator problems," IFAC-PapersOnLine,
%               vol. 54, no. 9, pp. 329–334, 2021, doi:
%               10.1016/j.ifacol.2021.06.090.
%               [2] M. Chen and D. Kressner, “Recursive blocked algorithms
%               for linear systems with Kronecker product structure,”
%               Numerical Algorithms, vol. 84, no. 3, pp. 1199–1216, Sep.
%               2019, doi: 10.1007/s11075-019-00797-5.
%               (https://www.epfl.ch/labs/anchp/index-html/software/misc/)
%
%
%   License: MIT
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%


if nargin < 6
    % solver = 'chen-kressner';
    solver = 'bartels-stewart';
    if nargin < 4
        E = [];
        if nargin < 3
            k = length(A);
        end
    end
end

if length(b) == 1
    solver = 'bartels-stewart'; % chen-kressner solver is broken for n=1 case
end

switch solver
    case 'bartels-stewart'
            %% k-way Bartels-Stewart Algorithm
            % We wish to solve the system ℒₖ(A)x=b. Let A=UTU'; then this is
            % equivalent to
            %       ℒₖ(A)x = (U⊗U⊗...⊗U)ℒₖ(T)(U⊗U⊗...⊗U)' x = b
            % Now, we can multiply by the conjugate transpose
            %       ℒ(T) (U⊗U⊗...⊗U)' x = (U⊗U⊗...⊗U)'b
            %   ≡   ℒₖ(T)|------ y ------| = |------ b̅ -----|
            % From here, we first solve this system for y via block back-
            % substitution. Then, we can multiply to get x:
            %       (U⊗U⊗...⊗U)' x = y   →   x = (U⊗U⊗...⊗U) y
            
            n = size(A{1},1);
            
            % Get Schur decomposition of A
            [U,T] = schur(A{1},'complex');
            
            % Replace linear system for x with linear system for y by
            % multiplying by (U⊗U⊗...⊗U)'
            b = kroneckerLeft(U',b);
            % [A{1:k}] = deal(T);
            
            if nargin < 5; UMU = sparse(n,n);
            else; UMU = U'*M*U; end % diagonal shift; not currently used
            
            %% Solve ℒₖ(T) y = b̅ via block back substitution
            % We will solve k-1 systems of dimension (n x n), starting
            % with the last one and then going backwards (backward
            % substitution). At each step, we need to form
            %   1) the coefficient matrix At
            %   2) the rhs vector, which is updated each time to include the
            %      previously computed quantities
            
            X = zeros(n,n^(k-1));
            B = reshape(b,n,n^(k-1));
            jRange = cell(1,k);
            for jj = n^(k-1):-1:1
                % jIndex is jj in k-digit base n; ex. [2 2 2] = 8, [2 2 1]  = 7, etc.
                jIndex = flip(tt_ind2sub(n*ones(1,k-1),jj));  
                colIdx = tt_sub2ind(n*ones(1,k-1),jIndex);
                
                %% 1) Construct At, the upper triangular coefficient matrix for the current unknowns
                % We know the coefficient is going to contain a
                % contribution from T, since (I⊗I⊗...⊗T) is a block
                % diagonal copying of T. However, we also need to
                % accumulate the contributions from (T⊗I⊗...⊗I),
                % (I⊗T⊗...⊗I), etc.
                diagA = 0;
                for i=1:k-1
                    diagA = diagA + T(jIndex(i),jIndex(i));
                end
                At = T + diagA*eye(n) + UMU;
                
                %% 2) Construct rhs, the b̅ updated with the already computed unknowns
                
                rhs = B(:,colIdx);
                
                %  Back-substitution steps
                for i=1:k-1    % this could be done by decreaseByOne as well
                    jRange{i} = (jIndex(i)+1):n;
                end
                
                for i=1:k-1
                    if (~isempty(jRange{i}))
                        shift = 1;
                        for j=1:(i-1)
                            shift = shift + (jIndex(j)-1)*n^(j-1);
                        end
                        for j=i:k-1
                            shift = shift + (jIndex(j)-1)*n^(j-1);
                        end
                        jIdx = shift + (jRange{i}-1)*n^(i-1);
                        
                        rhs = rhs - X(:,jIdx)*T(jIndex(i),jRange{i}).';
                    end
                end
                
                X(:,colIdx) = At\rhs;
                
            end
            
            % Now solve for x given y by multiplying by (U⊗U⊗...⊗U)
            x = real(kroneckerLeft(U,X(:)));
            
    case 'chen-kressner'
        try
            addpath('../tensor_recursive')
        catch
            disp("Download tensor_recursive package from https://www.epfl.ch/labs/anchp/index-html/software/misc/")
            error("tensor_recursive not found, please download it.")
        end
        
        n = length(A{1});
        B = reshape(b,n*ones(1,k));
        % if n^d < 2
        %     X = laplace_small(A, B);
        % else
        X = laplace_merge(A, B);
        % end
        x = real(X(:)); % real seems to be needed, not sure why
end

end


