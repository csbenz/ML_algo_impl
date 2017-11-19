function [X_hat] = reconstruct_pca(Y, A_p, Mu)
%RECONSTRUCT_PCA Reconstruct dataset to original dimensions from PCA
%   projection. In this function, the student should follow Eq. 6 from
%   Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o Y      : (p x M), Projected data set with N samples each being of dimension p.
%       o A_p    : (p x N), Projection Matrix.
%       o Mu     : (N x 1), Mean Vector of Dataset
%
%   output ----------------------------------------------------------------
%
%       o X_hat  : (N x M), reconstructed data set with M samples each being of dimension N.


% Auxiliary variables
[p, M] = size(Y);
[p, N] = size(A_p);

% Output Variables
X_hat = zeros(N,M);

% ====================== Implement Eq. 6 Here ====================== 
A_p_inv = (A_p.') * inv(A_p * (A_p.'));
X_hat = A_p_inv * Y + repmat(Mu,1,size(Y,2));

end