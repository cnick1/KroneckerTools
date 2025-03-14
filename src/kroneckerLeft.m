function Y = kroneckerLeft(M,B)
%kroneckerLeft Computes the product of repeated Kronecker products of M and a matrix B
%
%   Usage:  Y = kroneckerLeft(M,B)
%
%   Inputs:
%          M - cell array of matrices where M{k} has dimensions n1(k) x n2(k)
%              (optionally a constant matrix if M = M{1} = M{2} = ... = M{d})
%          B - matrix with dimensions prod(n2) x m
%
%   Output:
%          Y - matrix with dimensions prod(n1) x m
%
%   Description: We are interested in repeated Kronecker products like
%
%      Y = ( M₁ ⊗ M₂ ⊗ ... ⊗ Mᵈ ) B
%
%   If we were to evaluate this directly, the Kronecker product blows up
%   the dimensions and has very high memory requirements, in addition to
%   requiring many additional operations. Instead, we can reshape these
%   products as matrix-vector products and leverage BLAS operations [1].
%   The basis tool is the Kronecker-Vec relation
%
%       vec(ADB) = (Bᵀ ⊗ A) vec(D)
%
%   In the case that B is a vector, this applies directly (in the first
%   step), otherwise this applies column-wise, i.e.
%
%       [ vec(AD⁽¹⁾B) vec(AD⁽²⁾B) ... vec(AD⁽ᵈ⁾B) ] = (Bᵀ ⊗ A) D
%
%   Authors: Jeff Borggaard, Virginia Tech (Original author)
%            Nicholas Corbin, UC San Diego (Update to include E matrix)
%
%   References: [1] G. H. Golub and C. Van Loan, Matrix computations,
%                   Fourth. Johns Hopkins University Press, 2013.
%
%   License: MIT
%
%   Part of the KroneckerTools repository: github.com/cnick1/KroneckerTools
%%
[nB ,m] = size(B);

if (iscell(M))
    d = length(M);
    n1 = zeros(1,d); n2 = zeros(1,d);
    for i=1:d
        [n1(i),n2(i)] = size(M{i});
    end
    
    if ( d==2 )
        Y = zeros(n1(1)*n1(2),m);
        for j=1:m
            Y(:,j) = reshape(M{2}*reshape(B(:,j),n2(2),n2(1))*M{1}.',n1(1)*n1(2),1);
        end
        
    else
        n1Y = prod(n1);
        Y = zeros(n1Y,m);
        for j=1:m
            T = reshape(B(:,j),nB/n2(1),n2(1))*M{1}.';
            
            Z = zeros(prod(n1(2:end)),n1(1));
            for k=1:n1(1)
                Z(:,k) = kroneckerLeft(M(2:end),T(:,k));
            end
            Y(:,j) = Z(:);
        end
    end
    
else % assume M{i} = M, which is convienent for a change of variables.
    [nM,n] = size(M);
    [nB,m] = size(B);
    
    if ( n==nB )
        Y = M*B;
    elseif ( n^2==nB )
        Y = zeros(nM^2,m);
        for j=1:m
            Y(:,j) = reshape(M*reshape(B(:,j),n,n)*M.',nM^2,1);
        end
        
    else
        p = round(log(nB)/log(n));
        Y = zeros(nM^p,m);
        for j=1:m
            T = reshape(B(:,j),nB/n,n)*M.';
            
            Z = zeros(nM^(p-1),nM);
            for k=1:nM
                Z(:,k) = kroneckerLeft(M,T(:,k));
            end
            Y(:,j) = Z(:);
        end
    end
    
end

end

