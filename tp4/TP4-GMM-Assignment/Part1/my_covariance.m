function [ Sigma ] = my_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

% Output Variable
Sigma = 0;
[N,M] = size(X);

switch type 
    case 'full'
        X = bsxfun(@minus, X, X_bar);
        Sigma = (1 / (M - 1)) * (X * X.');
    case 'diag'
        Sigma = diag(diag(my_covariance( X, X_bar, 'full')));
    case 'iso'
        s = 0;
        for i=1:M
            n = norm(X(:,i) - X_bar)^2;
            s = s + n;
        end
        sigma_iso = (1 / (N * M)) * s;
        Sigma = sigma_iso * eye(N);
end

end

