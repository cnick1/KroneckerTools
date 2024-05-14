function [x] = KroneckerSumSolver(A,b,degree,M,solver)
%KroneckerSumSolver Efficiently solves a Kronecker sum system ℒ(A)x=b.
%
%   Usage: x = KroneckerSumSolver(A,b,degree)
%
%   Inputs:
%       A      - cell array containing the coefficient matrix of the linear 
%                system. Currently the implementation assumes each A is identical.
%                  • [Acell{1:degree}] = deal(A) assigns A to each entry
%                    in the cell array without duplicating memory usage.
%       b      - RHS vector of the linear system
%       degree - desired degree of the computed value function. A degree d
%                 energy function uses information from f,g,q up-to degree d-1.
%                 The default choice of d is lf+1, where lf is the degree of
%                 the drift.
%       M      - diagonal shift of the linear system.
%       solver - string specifying the solver. Options: 
%                  • 'bartels-stewart' - Jeff's N-way Bartels-Stewart
%                     direct solver from [1]. Described more below.
%                  • 'chen-kressner' - calls laplace_merge() solver from
%                     [2]. This performs better and is the default.
%   Output:
%       x       - cell array containing the polynomial value function coefficients
%
%  Background: Implements an N-Way version of the Bartels-Stewart algorithm
%  for a special Kronecker sum system (the special case all matrices are
%  the same) with an additional diagonal shift. 
%
%  Define the k-way Lyapunov Matrix 
%
%     ℒ_k(A) := (A⊗...⊗I_p) + (I_p⊗A⊗...⊗I_p) + ... + (I_p⊗...⊗A)
%
%  Then we have the linear system ℒ_k(A)x = b. Currently we assume the A
%  matrices are all the same, but they could be d different matrices of the
%  same size, so the function is written to take a cell array containing
%  the A matrices of size (n,n). The vectors b and x are size (n^d,1). M is
%  an nxn matrix that is added to block diagonal:  diag(M) = kron(eye(n^(d-1)),M)
%
%  For now, we assume each A{i} is the same (an N-Way Lyapunov equation) but
%  this can be relaxed by performing a Schur factorization to each A{i}
%  and performing the proper change of variables.
%
%  This is has been developed for the NLbalancing repository and its
%  functionality is consistent with the function KroneckerSumSolver that
%  was developed for the QQR problem (without the matrix M).
%
%  Author: Jeff Borggaard, Virginia Tech
%          Modified by Nick Corbin, UCSD
%
%
%` References: [1] J. Borggaard and L. Zietsman, "On approximating
%              polynomial-quadratic regulator problems," IFAC-PapersOnLine,
%              vol. 54, no. 9, pp. 329–334, 2021, doi:
%              10.1016/j.ifacol.2021.06.090.
%              [2] M. Chen and D. Kressner, “Recursive blocked algorithms
%              for linear systems with Kronecker product structure,”
%              Numerical Algorithms, vol. 84, no. 3, pp. 1199–1216, Sep.
%              2019, doi: 10.1007/s11075-019-00797-5.
%              (https://www.epfl.ch/labs/anchp/index-html/software/misc/)
%              
%
%  License: MIT
%
%  Part of the KroneckerTools repository: github.com/jborggaard/KroneckerTools
%%


if nargin < 5
    % solver = 'chen-kressner';
    solver = 'bartels-stewart';
end

if length(b) == 1
    solver = 'bartels-stewart'; % chen-kressner solver is broken for n=1 case
end

switch solver
    case 'bartels-stewart'
        n = size(A{1},1);


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

        X = zeros(n,n^(degree-1));
        B = reshape(b,n,n^(degree-1));

        jIndex = n*ones(1,degree);  jIndex(degree) = n+1;
        jRange = cell(1,degree);

        last = false;
        while( ~last )
            [jIndex,last] = decreaseByOne(jIndex,n);

            diagA = 0;
            for i=2:degree
                diagA = diagA + A{i}(jIndex(i),jIndex(i));
            end
            At = A{1} + diagA*eye(n) + UMU;

            colIdx = jIndex(2);
            for i=3:degree
                colIdx = colIdx + (jIndex(i)-1)*n^(i-2);
            end

            rhs = B(:,colIdx);

            %  Backsubstitution steps
            for i=2:degree    % this could be done by decreaseByOne as well
                jRange{i} = (jIndex(i)+1):n;
            end

            for i=2:degree
                if (~isempty(jRange{i}))
                    shift = 1;
                    for j=2:(i-1)
                        shift = shift + (jIndex(j)-1)*n^(j-2);
                    end
                    for j=(i+1):degree
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
        try
            addpath('../tensor_recursive')
        catch
            disp("Download tensor_recursive package from https://www.epfl.ch/labs/anchp/index-html/software/misc/")
            error("tensor_recursive not found, please download it.")
        end

        n = length(A{1}); d = length(A);
        B = reshape(b,n*ones(1,d));
        % if n^d < 2
        %     X = laplace_small(A, B);
        % else
        X = laplace_merge(A, B);
        % end
        x = real(X(:)); % real seems to be needed, not sure why
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
