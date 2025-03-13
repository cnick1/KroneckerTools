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


if nargin < 6
    % solver = 'chen-kressner';
    % solver = 'bartels-stewart';
    solver = 'bartels-stewart-updated';
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
        %   2) Now we need to solve ℒₖᵀᵉ(Tₐ)y=b̅. Since ℒₖᵀᵉ(Tₐ) is block triangular, we can solve
        %      this via block back substitution, where essentially as we solve for components of
        %      y, we move them to the right-hand-side.
        %
        %   3) From y, we can multiply by the right Schur vectors Z to get x:
        %       (Z⊗Z⊗...⊗Z)' x = y   →   x = (Z⊗Z⊗...⊗Z) y
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    1) Transform to triangular form    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get QZ decomposition of A,E
        [Ta,Te,Q,Z] = qz(full(A{1}),full(E),'complex');
        
        % Replace linear system for x with linear system for y by multiplying by (Q⊗Q⊗...⊗Q)
        b = kroneckerLeft(Q,b);
        matrices = cell(1,k-1); % this is just for coding purposes, not much extra memory is allocated by this
        [matrices{1:k-1}] = deal(Te);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Solve ℒₖᵀᵉ(Tₐ) y = b̅ via block back substitution. We will solve k-1 systems of dimension
        % (n x n), starting with the last one and then going backwards (backward substitution). At
        % each step, we need to form:
        %   a) the coefficient matrix At for the current unknowns
        %   2) the rhs vector, updated each time to include the previously computed quantities
        
        X = zeros(n,n^(k-1));
        B = reshape(b,n,n^(k-1));
        row_indices = n*ones(1,k-1); row_indices(end) = n+1;
        for colIdx = n^(k-1):-1:1
            % ~~~~~~~~~~~~~~~~~~~~    a) Construct At    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            % At is an n x n diagonal block of ℒₖᵀᵉ(Tₐ); ℒₖᵀᵉ(Tₐ), where Tₐ and Tₑ are both upper
            % triangular, is a  sum of a variety of permutations:
            %     (Tₐ⊗Tₑ⊗...⊗Tₑ) + (Tₑ⊗Tₐ⊗...⊗Tₑ) + ... + (Tₑ⊗Tₑ⊗...⊗Tₐ)
            % The ith diagonal blocks are therefore, by the definition of the Kronecker product,
            %     (Tₐ⊗Tₑ⊗...)ᵢᵢ Tₑ + (Tₑ⊗Tₐ⊗...)ᵢᵢ Tₑ + ... + (Tₑ⊗Tₑ⊗...)ᵢᵢ Tₐ
            % Notice that the only component that has "element-wise resolution" is the last one;
            % the other components are all some scalar multiple of Tₑ. What that means is that At
            % will always be α Tₐ + β Tₑ for some constant α and β; α and Tₐ and come from the
            % last permutation, and the other k-1 permutations determine β and Tₑ.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% How to get the entry (A⁽¹⁾⊗A⁽²⁾⊗...⊗A⁽ᵏ⁾)ᵢⱼ without forming the whole thing? %%%
            %%%     (A⁽¹⁾⊗A⁽²⁾⊗...⊗A⁽ᵏ⁾)ᵢⱼ = A⁽¹⁾ᵢ₁,ⱼ₁ * A⁽²⁾ᵢ₂,ⱼ₂ * ... * A⁽ᵏ⁾ᵢₖ,ⱼₖ          %%%
            %%% for some row indices i₁,i₂,...,iₖ and column indices j₁,j₂,...,jₖ. Getting the  %%%
            %%% indices turns out to be related to converting numbers to k-digit base n, and   %%%
            %%% can be done as follows:                                                        %%%
            %%%     row_indices = flip(tt_ind2sub(n*ones(1,k),i));                             %%%
            %%%     col_indices = flip(tt_ind2sub(n*ones(1,k),j));                             %%%
            %%% Evaluating product A⁽¹⁾ᵢ₁,ⱼ₁ * ... * A⁽ᵏ⁾ᵢₖ,ⱼₖ can be done with a simple loop:  %%%
            %%%     ABij = 1;                                                                  %%%
            %%%     for idx = 1:k                                                              %%%
            %%%         ABij = ABij * matrices{idx}(row_indices(idx), col_indices(idx));       %%%
            %%%     end                                                                        %%%
            %%% But we can actually do better; tt_ind2sub is somewhat expensive, and in the    %%%
            %%% end we are always incrementing k-digit base n numbers down by one every time.  %%%
            %%% So we can use the decreaseByOne function, and just initialize the variable     %%%
            %%% outside of the loop as one larger than the true starting value.                %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % At is the colIdxth coefficient along the diagonal; both α and β use the same indices,
            % just evaluating the product of different matrices
            % % Direct method: use ind2sub; however, this is a bit expensive every time
            % row_indices = flip(tt_ind2sub(n*ones(1,k-1),colIdx));
            % col_indices = row_indices; % col_indices = flip(tt_ind2sub(n*ones(1,k-1),colIdx)); % no reason to compute it twice
            % Efficient method: use decreaseByOne, identifying that we are incrementing one at a time
            row_indices = decreaseByOne2(row_indices,k-1,n);
            col_indices = row_indices; % no reason to compute it twice
            
            
            % ---------------------------------    Getting α    ----------------------------------
            % Here we just need to get (Tₑ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₐ
            alpha = 1;
            for idx = 1:k-1
                alpha = alpha * matrices{idx}(row_indices(idx), col_indices(idx));
            end
            
            % ---------------------------------    Getting β    ----------------------------------
            % Here we need to consider the k-1 other permutations
            % (Tₐ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₑ, (Tₑ⊗Tₐ⊗...⊗Tₑ)ᵢⱼ Tₑ, ..., (Tₑ⊗Tₑ⊗...⊗Tₐ)ᵢⱼ Tₑ
            beta = 0;
            for idx2 = 1:k-1
                % Can probably speed this up by not using matrices here, but rather doing two inner
                % loops with Te and a middle thing with Ta
                matrices{idx2} = Ta; % set the idx2th matrix to Ta
                betaTemp = 1;
                for idx = 1:k-1
                    betaTemp = betaTemp * matrices{idx}(row_indices(idx), col_indices(idx));
                end
                beta = beta + betaTemp;
                matrices{idx2} = Te; % reset the idx2th matrix to Te
            end
            
            At = alpha*Ta + beta*Te;
            
            % ~~~~~~~~~~~~~~~~~~~~~    b) Construct rhs vector    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            % This is the b̅ updated with the already computed unknowns
            % Start with original b̅
            rhs = B(:,colIdx);
            
            % Now update RHS by removing already computed things
            col_indices = n*ones(1,k-1); col_indices(end) = n+1;
            for idx3 = n^(k-1):-1:colIdx+1
                % The things we need to remove are n x n super-diagonal blocks of ℒₖᵀᵉ(Tₐ)
                % multiplied with n x 1 blocks of X that have already been computed. Getting these
                % blocks is similar to how we computed At, but now for off diagonal components.
                % Corresponding to a particular component of X will be a matrix Aod, which again
                % will have the form γ Tₐ + δ Tₑ for the same reasons as before. We just need to
                % compute γ and δ for each of the already computed X and corresponding Aod
                
                % % Direct method: use ind2sub; however, this is a bit expensive every time
                % % row_indices = flip(tt_ind2sub(n*ones(1,k-1),colIdx)); % already computed, no reason to compute it again
                % col_indices = flip(tt_ind2sub(n*ones(1,k-1),idx3));
                % Efficient method: use decreaseByOne, identifying that we are incrementing one at a time
                col_indices = decreaseByOne2(col_indices,k-1,n);
                
                
                % -------------------------------    Getting γ    --------------------------------
                % Here we just need to get (Tₑ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₐ
                gamma = 1;
                for idx = 1:k-1
                    gamma = gamma * matrices{idx}(row_indices(idx), col_indices(idx));
                end
                
                % -------------------------------    Getting δ    --------------------------------
                % Here we need to consider the k-1 other permutations
                % (Tₐ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₑ, (Tₑ⊗Tₐ⊗...⊗Tₑ)ᵢⱼ Tₑ, ..., (Tₑ⊗Tₑ⊗...⊗Tₐ)ᵢⱼ Tₑ
                delta = 0;
                for idx2 = 1:k-1
                    % Can probably speed this up by not using matrices here, but rather doing two inner
                    % loops with Te and a middle thing with Ta
                    % May also be able to vectorize
                    matrices{idx2} = Ta; % set the idx2th matrix to Ta
                    deltaTemp = 1;
                    for idx = 1:k-1
                        deltaTemp = deltaTemp * matrices{idx}(row_indices(idx), col_indices(idx));
                    end
                    delta = delta + deltaTemp;
                    matrices{idx2} = Te; % reset the idx2th matrix to Te
                end
                
                Aod = gamma*Ta + delta*Te;
                rhs = rhs - Aod*X(:,idx3);
            end
            
            % Solve for component of X
            X(:,colIdx) = At\rhs; % At is upper triangular
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%    3) Transform solution back    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Now solve for x given y by multiplying by (Z⊗Z⊗...⊗Z)
        x = real(kroneckerLeft(Z,X(:)));
        
    case 'bartels-stewart-updated'
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
        
        %%% 1) Transform to triangular form %%%
        % Get Schur decomposition of A
        [U,T] = schur(A{1},'complex');
        
        % Replace linear system for x with linear system for y by
        % multiplying by (U⊗U⊗...⊗U)'
        b = kroneckerLeft(U',b);
        
        if nargin < 5; UMU = sparse(n,n);
        else; UMU = U'*M*U; end % diagonal shift; not currently used
        
        %%% 2) Solve the transformed system %%%
        % Solve ℒₖ(T) y = b̅ via block back substitution
        % We will solve k-1 systems of dimension (n x n), moving things
        % we compute to the right-hand-side (back substitution). At
        % each step, we need to form
        %   a) the coefficient matrix At for the current unknowns
        %   b) the rhs vector, which is updated each time to include the
        %      previously computed quantities
        
        X = zeros(n,n^(k-1));
        B = reshape(b,n,n^(k-1));
        jRange = cell(1,k-1);
        jIndex = n*ones(1,k-1); jIndex(end) = n+1;
        for jj = n^(k-1):-1:1
            % jIndex is jj in k-1-digit base n; ex. [2 2 2] = 8, [2 2 1]  = 7, etc.
            % jIndex = flip(tt_ind2sub(n*ones(1,k-1),jj));
            jIndex = decreaseByOne2(jIndex,k-1,n);
            colIdx = tt_sub2ind(n*ones(1,k-1),jIndex); % colIdx doesn't exactly follow jj; I think this is because there are zero blocks that can be skipped when E = I
            
            %%% a) Construct At %%%
            % We know the coefficient is going to contain a
            % contribution from T, since (I⊗I⊗...⊗T) is a block
            % diagonal copying of T. However, we also need to
            % accumulate the contributions from (T⊗I⊗...⊗I),
            % (I⊗T⊗...⊗I), etc., which all appear along the diagonal
            % since the last matrix is always I for these permutations
            diagA = 0;
            for i=1:k-1
                diagA = diagA + T(jIndex(i),jIndex(i));
            end
            
            At = T + diagA*eye(n) + UMU;
            
            %%% b) Construct rhs vector %%%
            % This is the b̅ updated with the already computed unknowns
            % Start with original b̅
            rhs = B(:,colIdx);
            
            % Now update RHS by removing already computed things
            for i=1:k-1
                jRange{i} = (jIndex(i)+1):n;
            end
            
            for i=1:k-1
                if (~isempty(jRange{i}))
                    shift = 1;
                    for j=1:(i-1)
                        shift = shift + (jIndex(j)-1)*n^(j-1);
                    end
                    for j=(i+1):k-1
                        shift = shift + (jIndex(j)-1)*n^(j-1);
                    end
                    jIdx = shift + (jRange{i}-1)*n^(i-1);
                    
                    rhs = rhs - X(:,jIdx)*T(jIndex(i),jRange{i}).';
                end
            end
            
            % Solve for component of X
            X(:,colIdx) = At\rhs;
            
        end
        
        %%% 3) Transform solution back %%%
        % Now solve for x given y by multiplying by (U⊗U⊗...⊗U)
        x = real(kroneckerLeft(U,X(:)));
        
    case 'bartels-stewart'
        %% Jeff's Original k-way Bartels-Stewart Algorithm
        
        %  As a simplification, we assume all A{i} are the same size.  We furthermore
        %  assume they are all the same (so we can use the kroneckerLeft function as is)
        %  Both of these can be easily relaxed if an application arises.
        [U,T] = schur(A{1},'complex');
        b = kroneckerLeft(U',b);
        
        L = length(A);
        for l=1:L
            A{l} = T;
        end
        
        if nargin < 4
            UMU = sparse(n,n);
        else
            UMU = U'*M*U;
        end
        
        X = zeros(n,n^(k-1));
        B = reshape(b,n,n^(k-1));
        
        jIndex = n*ones(1,k);  jIndex(k) = n+1;
        jRange = cell(1,k);
        
        last = false;
        while( ~last )
            [jIndex,last] = decreaseByOne(jIndex,n);
            
            diagA = 0;
            for i=2:k
                diagA = diagA + A{i}(jIndex(i),jIndex(i));
            end
            At = A{1} + diagA*eye(n) + UMU;
            
            colIdx = jIndex(2);
            for i=3:k
                colIdx = colIdx + (jIndex(i)-1)*n^(i-2);
            end
            
            rhs = B(:,colIdx);
            
            %  Backsubstitution steps
            for i=2:k    % this could be done by decreaseByOne as well
                jRange{i} = (jIndex(i)+1):n;
            end
            
            for i=2:k
                if (~isempty(jRange{i}))
                    shift = 1;
                    for j=2:(i-1)
                        shift = shift + (jIndex(j)-1)*n^(j-2);
                    end
                    for j=(i+1):k
                        shift = shift + (jIndex(j)-1)*n^(j-2);
                    end
                    jIdx = shift + (jRange{i}-1)*n^(i-2);
                    
                    rhs = rhs - X(:,jIdx)*A{i}(jIndex(i),jRange{i}).';
                end
            end
            
            X(:,colIdx) = At\rhs;
            
        end
        
        x = X(:);
        x = real(kroneckerLeft(U,x));
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

