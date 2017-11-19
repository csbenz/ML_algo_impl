function [ V, L, Mu ] = my_pca( X )
%MY_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm following Eq.1, 2 and 3 of Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o V      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
V  = zeros(M,M);
L  = zeros(M,M);
Mu = zeros(N,1);

% ====================== Implement Eq. 1 Here ====================== 
X = bsxfun(@minus, X, mean(X, 2));

% ====================== Implement Eq.2 Here ======================
%X_T = transpose(X);
C = X*(X.')/(M-1);

% ====================== Implement Eq.3 Here ======================
[V,L] = eig(C);

% =================== Sort Eigenvectors wrt. EigenValues ==========
[sorted_l, index] = sort(diag(L), 'descend');

V=-V(:,index); 
L = diag(sorted_l); 

end

