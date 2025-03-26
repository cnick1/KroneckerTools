clear; clc;
n = 5; k = 3;
% rng(0,'twister')

% Generate a random A
A = rand(n,n);

% Generate a random E
E = rand(n,n);

% Generate a random SYMMETRIC solution xe
xe = kronMonomialSymmetrize(rand(n^k,1), n, k);

% Construct b so that xe is the solution to ℒₖᴱ(A)x=b
b = LyapProduct(A,xe,k,E);

%% Solve with solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%    1) Transform to triangular form    %%%%%%%%%%%%%%%%%%%%%%%%%%
% Get QZ decomposition of A,E
[Ta,Te,Q,Z] = qz(A,E,'complex');

% Replace linear system for x with linear system for y by multiplying by (Q⊗Q⊗...⊗Q)
b = kroneckerLeft(Q,b);

% b = LyapProduct(Ta,ones(n^k,1),k,Te);
b = reshape(b,n*ones(1,k));

%%%%%%%%%%%%%%%%%%%%%%%%%%    2) Solve the transformed system    %%%%%%%%%%%%%%%%%%%%%%%%%%
if k == 3
    X1 = lyap3c(Ta,Te,b);
elseif k == 4
    X1 = lyap4c(Ta,Te,b);
end
X2 = lyapkc(Ta,Te,b,k);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%    3) Transform solution back    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now solve for x given y by multiplying by (Z⊗Z⊗...⊗Z)
x1 = real(kroneckerLeft(Z,X1(:)));
x2 = real(kroneckerLeft(Z,X2(:)));

%%
fprintf('The test for n=%i, k=%i has error %g\n',n,k,norm(x1-xe));
fprintf('The test for n=%i, k=%i has error %g\n',n,k,norm(x2-xe));


