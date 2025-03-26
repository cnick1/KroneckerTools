%%

clear;clc;
% rng(0,'twister')
n=3;
A = rand(n);
E = rand(n);
X = ones(n,n);


[A,E,Q,Z] = qz(A,E,'complex');

% AA = Q*A*Z;
% EE = Q*E*Z;
% Y = Q*A*Q';
Y = (A'*X*E+E'*X*A);

lyap(A',-Y,[],E')
lyap2c(A,E,Y,true)

%% k=3 case forwards
addpath('src')
addpath('test')
clear;clc;
rng(0,'twister')
n=3;
A = triu(rand(n));
E = triu(rand(n));
X = ones(n,n,n);

Y = reshape(LyapProduct(A',X(:),3,E'),n,n,n);
tic
X1 = lyap3c(A,E,Y,true);
toc
tic
X2 = lyapkc(A, E, Y);
toc


%% k=3 case backwards
addpath('src')
addpath('test')
clear;clc;
rng(0,'twister')
n=30;
A = triu(rand(n));
E = triu(rand(n));
X = ones(n,n,n);

Y = reshape(LyapProduct(A,X(:),3,E),n,n,n);

tic
X1 = lyap3c(A,E,Y);
toc
tic
X2 = lyapkc(A,E,Y);
toc

%% k=4 case forwards
addpath('src')
addpath('test')
clear;clc;
rng(0,'twister')
n=3;
A = triu(rand(n));
E = triu(rand(n));
X = ones(n,n,n,n);

Y = reshape(LyapProduct(A',X(:),4,E'),n,n,n,n);
tic
X1 = lyap4c(A,E,Y,true);
toc
tic
X2 = lyapkc(A,E,Y);
toc


%% k=4 case backwards
addpath('src')
addpath('test')
clear;clc;
rng(0,'twister')
n=3;
A = triu(rand(n));
E = triu(rand(n));
X = ones(n,n,n,n);

Y = reshape(LyapProduct(A,X(:),4,E),n,n,n,n);
tic
X1 = lyap4c(A,E,Y);
toc
tic
X2 = lyapkc(A,E,Y);
toc

%%
a = triu(sym('a', [n, n])); e = triu(sym('e',[n, n])); 
kron(kron(a,e),e) + kron(kron(e,a),e) + kron(kron(e,e),a)