function [X] = lyapkc(A, E, Y, k)
%lyapkc Solves the k-way reduced continuous-time generalized Lyapunov equation
%
%   Usage: X = lyapkc(A,E,Y,k)
%
%   Inputs:
%       A      - coefficient matrix in triangular form (n x n)
%       E      - mass matrix in triangular form        (n x n)
%       Y      - symmetric right-hand-side matrix      (n x ... x n)
%
%   Output:
%       X      - solution to x = ℒₖᴱ(A)⁻¹vec(Y)
%
%   Background: Define the k-way Lyapunov Matrix
%
%             |--------------------------------------k terms----------------|
%   ℒₖᴱ(A) = ( A ⊗ ... ⊗ E  +  E ⊗ A ⊗ ... ⊗ E  +  ...  +  E ⊗ ... ⊗ A )
%             |--k factors--|   |---k factors---|      ...    |--k factors--|
%
%   Then the linear system ℒₖᴱ(A)x = b is a tensor version of the Lyapunov
%   equation, where x = vec(X) and b = vec(Y). For example,
%
%       ℒ₂ᴱ(A)x = b  ⟷  (A⊗E + E⊗A)vec(X) = vec(Y)  ⟷  AXE' + EXA' = Y
%
%   This function implements a block-based generalized Bartels-Stewart
%   algorithm, with computational complexity O(knᵏ⁺²) (I need to confirm
%   this). The Bartels-Stewart algorithm consists of 3 steps:
%       1) Transform to the triangular form
%       2) Solve the transformed system
%       3) Transform the solution back
%   These are done as follows:
%       1) Let A = Q'TₐZ' and E = Q'TₑZ' be the QZ decomposition of A,E;
%          then the k-way generalized Lyapunov equation is equivalent to
%              ℒₖᴱ(A)x = (Q⊗Q⊗...⊗Q)'ℒₖᵀᵉ(Tₐ)(Z⊗Z⊗...⊗Z)' x = b
%          We can multiply by the left Schur vectors Q to get the
%          triangular form (sometimes called reduced system)
%              ℒₖᵀᵉ(Tₐ) (Z⊗Z⊗...⊗Z)' x = (Q⊗Q⊗...⊗Q)b
%                       |------ y ------|  |------ b̅ -----|
%
%       2) Now we need to solve ℒₖᵀᵉ(Tₐ)y=b̅. Since ℒₖᵀᵉ(Tₐ) is block
%          triangular, we can solve this via block back substitution, where
%          essentially as we solve for components of y, we move them to the
%          right-hand-side.
%
%       3) From y, we can multiply by the right Schur vectors Z to get x:
%              (Z⊗Z⊗...⊗Z)' x = y   →   x = (Z⊗Z⊗...⊗Z) y
%
%   This function specifically focuses on step 2, which is the main
%   challenge, since the other steps just involve transforming by Q and Z.
%
%   Authors: Nicholas Corbin, UC San Diego
%
%`  References: [1]
%
%   License: MIT
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%
vec = @(x) x(:);
n = size(A,1);
if nargin < 4
    k = ndims(Y);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve ℒₖᵀᵉ(Tₐ) y = b̅ via block back substitution. We will solve k-1 systems of dimension
% (n x n). In theory we could start with the last one and then go backwards (backward
% substitution). However, there is sparsity we need to exploit, so in reality we are going
% to do a somewhat strange thing where we jump around a lot. I think the best way to do this
% is by incrementing in k-digit base n. I'll do a for loop with a dummy index, and colIdx is
% going to be calculated each time.
% At each step, we need to form:
%   a) the coefficient matrix At for the current unknowns
%   2) the rhs vector, updated each time to include the previously computed quantities

X = zeros(n,n^(k-1));
B = reshape(Y,n,n^(k-1));
km1ones = ones(1,k-1);
row_indices = n*km1ones; row_indices(1) = n+1;
mult = [1 n.^(1:k-2)]'; % used repeatedly when converting between k-digit base n and decimal
% Now, in principle, we could do back-substitution with colIdx = n^(k-1):-1:1; if we do that
% though, we are neglecting the fractal nature of the problem and it is hard to take
% advantage of the sparsity. What we really want to do is jump in the following fashion:
%   colIdx_range = vec(permute(reshape(n^(k-1):-1:1,n*ones(1,k-1)),k-1:-1:1)).';
% However, this is an n^(k-1) x 1 vector, which is quite expensive to store, so we would
% rather not form the entire thing to iterate (e.g. for i=1:10^100 doesn't store 1:10^100)
% Those iterations are more easily performed on the subscript level, so the for loop will use
% a dummy variable and inside we will iterate row_indices using the decreaseByOne function
for itNum = 1:n^(k-1)
    % ====================    a) Construct At    =========================================
    % At is the colIdxth coefficient along the diagonal; both α and β use the same indices,
    % just evaluating the product of different matrices
    row_indices = decreaseByOneFlipped(row_indices,k-1,n,km1ones); % decreaseByOneFlipped iterates with the jumps conveniently
    col_indices = row_indices; % no reason to compute it twice
    % colIdx = tt_sub2ind(n*ones(1,k-1),flip(row_indices));
    colIdx = (flip(row_indices) - 1) * mult + 1; % essentially tt_sub2ind

    % --------------------    Getting α    -----------------------------------------------
    % Here we just need to get (Tₑ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₐ
    ind = row_indices + (col_indices - 1).*n; % basically sub2ind
    % fprintf('\nDiagonal ind:      '), fprintf('%i ',ind), fprintf('\n')
    Tas = A(ind); Tes = E(ind);
    alpha = prod(Tes);

    % --------------------    Getting β    -----------------------------------------------
    % Here we need to consider the k-1 other permutations
    % (Tₐ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₑ, (Tₑ⊗Tₐ⊗...⊗Tₑ)ᵢⱼ Tₑ, ..., (Tₑ⊗Tₑ⊗...⊗Tₐ)ᵢⱼ Tₑ
    beta = sum(Tas .* alpha./Tes); % black magic; basically alpha is prod(Tes), but we want to swap the idx2th position with the element from Ta instead

    At = alpha*A + beta*E;

    % ====================    b) Construct rhs vector    =================================
    % This is the b̅ updated with the already computed unknowns
    % Start with original b̅, then update with already computed things (back-substitution)
    rhs = B(:,colIdx);

    % For the back-substitution, in principle we could to regular triangular
    % back-substitution, i.e. index from the last column backwards to the diagonal index
    % like idx3 = n^(k-1):-1:colIdx+1; however, this neglects a massive amount of sparsity
    % available due to the triangular fractal nature of the problem. Instead, we want to
    % trace back through the colIdxs that we have already computed... BUT we want to still
    % skip things that are "below" the diagonal, since they are zero. So we want something
    % that works like decreaseByOneFlipped from before, but that takes into account the
    % diagonal index to basically skip certain components that are unimportant. The way
    % I am thinking to do this is with an additional argument to decreaseByOneFlipped, the
    % current row_indices. So idx3 will be a dummy index again.
    % col_indices = n*ones(1,k-1); col_indices(1) = n+1;
    % for idx3 = n^(k-1):-1:colIdx+1
    % idx3_range = sort(colIdx_range(1:itNum),'descend'); % order doesn't matter, but we need to not go lower than col_Idx -> corresponds to row_indices
    % idx3_range(find(idx3_range == colIdx):end) = []; % makes sure we don't go lower than col_idx
    % for idx3 = 1:itNum-1
    backSubColIdx = n^(k-1); col_indices = n*km1ones;
    while backSubColIdx ~= colIdx
        % -------------------------------    Getting γ    --------------------------------
        % Here we just need to get (Tₑ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₐ
        ind = row_indices + (col_indices - 1).*n; % basically sub2ind
        % fprintf('   Back-sub ind:   '), fprintf('%i ',ind), fprintf('\n')
        Tas = A(ind); Tes = E(ind);
        gamma = prod(Tes);

        % -------------------------------    Getting δ    --------------------------------
        % Here we need to consider the k-1 other permutations
        % (Tₐ⊗Tₑ⊗...⊗Tₑ)ᵢⱼ Tₑ, (Tₑ⊗Tₐ⊗...⊗Tₑ)ᵢⱼ Tₑ, ..., (Tₑ⊗Tₑ⊗...⊗Tₐ)ᵢⱼ Tₑ
        delta = sum(Tas .* gamma./Tes); % black magic; basically gamma is prod(Tes), but we want to swap the idx2th position with the element from Ta instead

        Aod = gamma*A + delta*E;
        rhs = rhs - Aod*X(:,backSubColIdx); % It seems like Ta*X and Te*X are frequently computed (and then inverted) modulo a scalar multiple; can I pre-compute or overwrite something and then just deal with scalar multiplication of those vectors, rather than having to do all this matrix-vector multiplication

        col_indices = decreaseByOneFlipped(col_indices,k-1,n,row_indices);
        % backSubColIdx = tt_sub2ind(n*ones(1,k-1),flip(col_indices));
        backSubColIdx = (flip(col_indices) - 1) * mult + 1; % essentially tt_sub2ind
    end

    % Solve for component of X
    X(:,colIdx) = At\rhs; % At is upper triangular
end

X = reshape(X,n*ones(1,k));

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
