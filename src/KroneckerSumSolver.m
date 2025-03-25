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
%                  • 'generalized-bartels-stewart' - Nicks's generalized
%                     N-way Bartels-Stewart that includes E matrix. If an E
%                     matrix is included, this solver will be selected.
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
vec = @(x) x(:);

if nargin < 6 || isempty(solver)
    % solver = 'chen-kressner';
    solver = 'bartels-stewart';
    % solver = 'bartels-stewart-jeffs';
    % solver = 'generalized-bartels-stewart';
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

if ~isempty(E)
    solver = 'generalized-bartels-stewart';
end

n = size(A{1},1);
switch solver
    case 'generalized-bartels-stewart'
        %% Generalized k-way Bartels-Stewart Algorithm
        % We wish to solve the system ℒₖᴱ(A)x=b, which is a generalized k-way Lyapunov equation
        % (has a mass matrix E). Bartels-Stewart algorithm for solving consists of 3 steps:
        %   1) Transform to the triangular form
        %   2) Solve the transformed system
        %   3) Transform the solution back
        % These are done as follows:
        %   1) Let A = Q'TₐZ' and E = Q'TₑZ' be the QZ decomposition of A,E; then the k-way
        %      generalized Lyapunov equation is equivalent to
        %         ℒₖᴱ(A)x = (Q⊗Q⊗...⊗Q)'ℒₖᵀᵉ(Tₐ)(Z⊗Z⊗...⊗Z)' x = b
        %      We can multiply by the left Schur vectors Q to get the triangular form (sometimes
        %      called reduced system)
        %         ℒₖᵀᵉ(Tₐ) (Z⊗Z⊗...⊗Z)' x = (Q⊗Q⊗...⊗Q)b
        %                  |------ y ------|  |------ b̅ -----|
        %
        %   2) Now we need to solve ℒₖᵀᵉ(Tₐ)y=b̅. This is the reduced generalized Lyapunov equation,
        %      which means that it has a special triangular structure that permits a special type of
        %      back substitution. This is performed by an external function.
        %
        %   3) From y, we can multiply by the right Schur vectors Z to get x:
        %       (Z⊗Z⊗...⊗Z)' x = y   →   x = (Z⊗Z⊗...⊗Z) y
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    1) Transform to triangular form    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get QZ decomposition of A,E
        [Ta,Te,Q,Z] = qz(full(A{1}),full(E),'complex');
        
        % Replace linear system for x with linear system for y by multiplying by (Q⊗Q⊗...⊗Q)
        b = kroneckerLeft(Q,b);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
        b = reshape(b,n*ones(1,k));
        if k == 2
            X = lyap(Ta,-b,[],Te);
        elseif k == 3
            X = lyap3c(Ta,Te,b);
        elseif k == 4
            X = lyap4c(Ta,Te,b);
        else
            warning("lyapkc: this solver is inefficient and may take a long time")
            X = lyapkc(Ta,Te,b,k);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%    3) Transform solution back    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Now solve for x given y by multiplying by (Z⊗Z⊗...⊗Z)
        x = real(kroneckerLeft(Z,X(:)));
        
    case 'bartels-stewart'
        %% k-way Bartels-Stewart Algorithm
        % We wish to solve the system ℒₖ(A)x=b. Bartels-Stewart
        % algorithm for solving consists of 3 steps:
        %   1) Transform to the triangular form
        %   2) Solve the transformed system
        %   3) Transform the solution back
        % These are done as follows:
        %   1) Let A=UTU' be the Schur decomposition of A; then the k-way
        % Lyapunov equation is equivalent to
        %       ℒₖ(A)x = (U⊗U⊗...⊗U)ℒₖ(T)(U⊗U⊗...⊗U)' x = b
        % We can multiply by the conjugate transpose of the Schur vectors
        % to get the triangular form (sometimes called reduced system)
        %       ℒ(T) (U⊗U⊗...⊗U)' x = (U⊗U⊗...⊗U)'b
        %   ≡   ℒₖ(T)|------ y ------| = |------ b̅ -----|
        %
        %   2) Now we need to solve ℒₖ(T)y=b̅. Since ℒₖ(T) is block
        %   triangular, we can solve this via block back substitution,
        %   where essentially as we solve for components of y, we move
        %   them to the right-hand-side.
        %
        %   3) From y, we can multiply by the Schur vectors to get x:
        %       (U⊗U⊗...⊗U)' x = y   →   x = (U⊗U⊗...⊗U) y
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    1) Transform to triangular form    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get Schur decomposition of A
        [U,T] = schur(A{1},'complex'); I = eye(n);
        
        % Replace linear system for x with linear system for y by multiplying by (U⊗U⊗...⊗U)'
        b = kroneckerLeft(U',b);
        
        if nargin < 5 || isempty(M); UMU = sparse(n,n);
        else; UMU = U'*M*U; end % diagonal shift; not currently used
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Solve ℒₖ(T) y = b̅ via block back substitution
        % We will solve k-1 systems of dimension (n x n), moving things
        % we compute to the right-hand-side (back substitution). At
        % each step, we need to form
        %   a) the coefficient matrix At for the current unknowns
        %   b) the rhs vector, which is updated each time to include the
        %      previously computed quantities
        
        X = zeros(n,n^(k-1));
        B = reshape(b,n,n^(k-1));
        km1ones = ones(1,k-1);
        row_indices = n*km1ones; row_indices(end) = n+1;
        mult = [1 n.^(1:k-2)]'; % used repeatedly when converting between k-digit base n and decimal
        for itNum = n^(k-1):-1:1
            % ====================    a) Construct At    =========================================
            % row_indices is itNum in k-1-digit base n; ex. [2 2 2] = 8, [2 2 1]  = 7, etc.
            row_indices = decreaseByOneFlipped(row_indices,k-1,n,km1ones); % decreaseByOneFlipped iterates with the jumps conveniently
            colIdx = (flip(row_indices) - 1) * mult + 1; % essentially tt_sub2ind
            
            % --------------------    Getting α    -----------------------------------------------
            % Here we just need to get (I⊗I⊗...⊗I)ᵢᵢ T
            % α=1 since Tₑ=I
            
            % --------------------    Getting β    -----------------------------------------------
            % Here we need to consider the k-1 other permutations
            % (T⊗I⊗...⊗I)ᵢᵢ I, (I⊗T⊗...⊗I)ᵢᵢ I, ..., (I⊗I⊗...⊗T)ᵢᵢ I
            ind = row_indices + (row_indices - 1).*n; % basically sub2ind
            beta = sum(T(ind)); % black magic
            
            At = T + beta*I + UMU;
            
            % ====================    b) Construct rhs vector    =================================
            % This is the b̅ updated with the already computed unknowns
            % Start with original b̅, then update with already computed things (back-substitution)
            rhs = B(:,colIdx);
            
            % Now update RHS by removing already computed things
            % Method 2: mine
            backSubColIdx = n^(k-1); col_indices = n*km1ones;
            while backSubColIdx ~= colIdx
                % -------------------------------    Getting γ    --------------------------------
                % Here we just need to get (Tₑ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₐ
                % γ = 0 since Tₑ=I
                ind = row_indices + (col_indices - 1).*n; % basically sub2ind
                % fprintf('   Back-sub ind:   '), fprintf('%i ',ind), fprintf('\n')
                % Tas = T(ind);
                
                % -------------------------------    Getting δ    --------------------------------
                % Here we need to consider the k-1 other permutations
                % (Tₐ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₑ, (Tₑ⊗Tₐ⊗...⊗Tₑ)ᵢⱼ Tₑ, ..., (Tₑ⊗Tₑ⊗...⊗Tₐ)ᵢⱼ Tₑ
                delta = 0; % black magic; basically gamma is prod(Tes), but we want to swap the idx2th position with the element from Ta instead
                for i=1:k-1
                    delta = delta + sum( ...
                        prod( ...
                        [T(ind(i)) I(ind([1:(i-1) (i+1):k-1]))]...
                        ));
                end
                
                rhs = rhs - delta*X(:,backSubColIdx); % It seems like Ta*X and Te*X are frequently computed (and then inverted) modulo a scalar multiple; can I pre-compute or overwrite something and then just deal with scalar multiplication of those vectors, rather than having to do all this matrix-vector multiplication
                
                col_indices = decreaseByOneFlipped(col_indices,k-1,n,row_indices);
                % backSubColIdx = tt_sub2ind(n*ones(1,k-1),flip(col_indices));
                backSubColIdx = (flip(col_indices) - 1) * mult + 1; % essentially tt_sub2ind
            end
            
            % Solve for component of X
            X(:,colIdx) = At\rhs;
            
        end
        
        %%% 3) Transform solution back %%%
        % Now solve for x given y by multiplying by (U⊗U⊗...⊗U)
        x = real(kroneckerLeft(U,X(:)));
        
    case 'bartels-stewart-jeffs'
        %% k-way Bartels-Stewart Algorithm
        % We wish to solve the system ℒₖ(A)x=b. Bartels-Stewart
        % algorithm for solving consists of 3 steps:
        %   1) Transform to the triangular form
        %   2) Solve the transformed system
        %   3) Transform the solution back
        % These are done as follows:
        %   1) Let A=UTU' be the Schur decomposition of A; then the k-way
        % Lyapunov equation is equivalent to
        %       ℒₖ(A)x = (U⊗U⊗...⊗U)ℒₖ(T)(U⊗U⊗...⊗U)' x = b
        % We can multiply by the conjugate transpose of the Schur vectors
        % to get the triangular form (sometimes called reduced system)
        %       ℒ(T) (U⊗U⊗...⊗U)' x = (U⊗U⊗...⊗U)'b
        %   ≡   ℒₖ(T)|------ y ------| = |------ b̅ -----|
        %
        %   2) Now we need to solve ℒₖ(T)y=b̅. Since ℒₖ(T) is block
        %   triangular, we can solve this via block back substitution,
        %   where essentially as we solve for components of y, we move
        %   them to the right-hand-side.
        %
        %   3) From y, we can multiply by the Schur vectors to get x:
        %       (U⊗U⊗...⊗U)' x = y   →   x = (U⊗U⊗...⊗U) y
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    1) Transform to triangular form    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get Schur decomposition of A
        [U,T] = schur(A{1},'complex'); I = eye(n);
        
        % Replace linear system for x with linear system for y by multiplying by (U⊗U⊗...⊗U)'
        b = kroneckerLeft(U',b);
        
        if nargin < 5 || isempty(M); UMU = sparse(n,n);
        else; UMU = U'*M*U; end % diagonal shift; not currently used
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Solve ℒₖ(T) y = b̅ via block back substitution
        % We will solve k-1 systems of dimension (n x n), moving things
        % we compute to the right-hand-side (back substitution). At
        % each step, we need to form
        %   a) the coefficient matrix At for the current unknowns
        %   b) the rhs vector, which is updated each time to include the
        %      previously computed quantities
        
        X = zeros(n,n^(k-1));
        B = reshape(b,n,n^(k-1));
        km1ones = ones(1,k-1);
        row_indices = n*km1ones; row_indices(end) = n+1;
        mult = [1 n.^(1:k-2)]'; % used repeatedly when converting between k-digit base n and decimal
        jRange = cell(1,k-1);
        for itNum = n^(k-1):-1:1
            % ====================    a) Construct At    =========================================
            % row_indices is itNum in k-1-digit base n; ex. [2 2 2] = 8, [2 2 1]  = 7, etc.
            % row_indices = flip(tt_ind2sub(n*ones(1,k-1),itNum));
            row_indices = decreaseByOne2(row_indices,k-1,n);
            % colIdx = tt_sub2ind(n*ones(1,k-1),row_indices);
            colIdx = (row_indices - 1) * mult + 1; % essentially tt_sub2ind
            
            % --------------------    Getting α    -----------------------------------------------
            % Here we just need to get (I⊗I⊗...⊗I)ᵢⱼ T
            % α=1 since Tₑ=I
            
            % --------------------    Getting β    -----------------------------------------------
            % Here we need to consider the k-1 other permutations
            % (T⊗I⊗...⊗I)ᵢⱼ I, (I⊗T⊗...⊗I)ᵢⱼ I, ..., (I⊗I⊗...⊗T)ᵢⱼ I
            % Method 1: Loop                                 ~ 36-37 sec
            % beta = 0;
            % for i=1:k-1
            %     beta = beta + T(row_indices(i),row_indices(i));
            % end
            % Method 2: Vectorized                           ~ 36-37 sec
            ind = row_indices + (row_indices - 1).*n; % basically sub2ind
            beta = sum(T(ind)); % black magic
            
            
            At = T + beta*I + UMU;
            
            % ====================    b) Construct rhs vector    =================================
            % This is the b̅ updated with the already computed unknowns
            % Start with original b̅, then update with already computed things (back-substitution)
            rhs = B(:,colIdx);
            
            % Now update RHS by removing already computed things
            % Method 1: Jeff's
            % for i=1:k-1
            %     jRange{i} = (row_indices(i)+1):n;
            % end
            % jRange = arrayfun(@(x) (x + 1):n, row_indices, 'UniformOutput', false);
            %
            %
            % for i=1:k-1
            %     if (~isempty(jRange{i}))
            %         shift = (row_indices([1:(i-1) (i+1):k-1]) - 1) * mult([1:(i-1) (i+1):k-1]) + 1; % essentially tt_sub2ind
            %
            %         backSubColIdx = shift + (jRange{i}-1)*n^(i-1); % becomes a vector of same dimension as jRange{i}
            %
            %         deltas = T(row_indices(i),jRange{i}).';
            %
            %         rhs = rhs - X(:,backSubColIdx)*deltas;
            %     end
            % end
            
            % Method 2: Mix of Jeff's and mine
            for i=1:k-1
                colIdxRange = (row_indices(i)+1):n;
                
                if (~isempty(colIdxRange))
                    shift = (row_indices([1:(i-1) (i+1):k-1]) - 1) * mult([1:(i-1) (i+1):k-1]) + 1; % essentially tt_sub2ind
                    
                    backSubColIdx = shift + (colIdxRange-1)*n^(i-1); % becomes a vector of same dimension as jRange{i}
                    
                    deltas = T(row_indices(i),colIdxRange).';
                    
                    rhs = rhs - X(:,backSubColIdx)*deltas;
                end
                
            end
            
            % Solve for component of X
            X(:,colIdx) = At\rhs;
            
        end
        
        %%% 3) Transform solution back %%%
        % Now solve for x given y by multiplying by (U⊗U⊗...⊗U)
        x = real(kroneckerLeft(U,X(:)));
        
    case 'chen-kressner'
        %% Chen & Kressner recursive blocked algorithm
        % The back substitution steps in Bartels-Stewart uses primarily level 2
        % BLAS operations (matrix-vector multiplications). This paper
        % proposes an algorithm that better leverages level 3 BLAS
        % operations (is more memory efficient)
        try
            addpath('../tensor_recursive')
        catch
            disp("Download tensor_recursive package from https://www.epfl.ch/labs/anchp/index-html/software/misc/")
            error("tensor_recursive not found, please add it to path.")
        end
        
        B = reshape(b,n*ones(1,k));
        % if n^d < 2
        %     X = laplace_small(A, B);
        % else
        X = laplace_merge(A, B);
        % end
        x = real(X(:)); % real needed due to use of complex Schur decomposition
end

end


function [jIndex,last] = decreaseByOne(jIndex,n)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dp1 = length(jIndex);
d   = dp1-1;

if (jIndex(dp1)==1)
    jIndex(dp1) = n;
    jIndex(1:d) = decreaseByOne(jIndex(1:d),n);
    
else
    jIndex(dp1) = jIndex(dp1)-1;
    
end

last = norm(jIndex(2:end)-ones(1,d))==0;

end

function [col_indices] = decreaseColIdxsByOne2(col_indices,row_indices,k,n)
%decreaseByOne2 Decrease jIndex, which is a k-digit number in base n, by 1
%   Detailed explanation goes here

if col_indices(k) == row_indices(k)
    col_indices(k) = n;
    col_indices(1:k-1) = decreaseColIdxsByOne2(col_indices(1:k-1),row_indices(1:k-1),k-1,n);
else
    col_indices(k) = col_indices(k)-1;
end

end

function [jIndex] = decreaseByOne2(jIndex,k,n)
%decreaseByOne2 Decrease jIndex, which is a k-digit number in base n, by 1
%   Detailed explanation goes here

if jIndex(k) == 1
    jIndex(k) = n;
    jIndex(1:k-1) = decreaseByOne2(jIndex(1:k-1),k-1,n);
else
    jIndex(k) = jIndex(k)-1;
end

end

function [indices_1] = decreaseByOneFlipped(indices_1,k,n,indices_2)
%decreaseByOneFlipped Decrease indices_1, which is a k-digit number in base n, by 1
%   Detailed explanation goes here

if indices_1(1) == indices_2(1)
    indices_1(1) = n;
    indices_1(2:k) = decreaseByOneFlipped(indices_1(2:k),k-1,n,indices_2(2:k));
else
    indices_1(1) = indices_1(1)-1;
end

end

function ndx = sub2ind(siz,v1,v2)
%SUB2IND Linear indices from multiple subscripts.
%   SUB2IND is used to determine the equivalent single index
%   corresponding to a given set of subscript values.
%
%   IND = SUB2IND(SIZ,I,J) returns the linear indices equivalent to the
%   row and column subscripts in the arrays I and J for a matrix of
%   size SIZ.
%
%   See also IND2SUB.
%
%   This is a stripped down version of Matlab's builtin sub2ind to remove
%   all the unnecessary overhead.
%   Copyright 1984-2015 The MathWorks, Inc.



ndx = double(v1);
if numOfIndInput >= 2
    %Compute linear indices
    ndx = ndx + (double(v2) - 1).*siz(1);
end
ndx = v1 + (v2 - 1).*siz(1);

end
