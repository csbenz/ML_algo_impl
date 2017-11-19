function [Mu] =  kmeans_init(X, k, init)
%KMEANS_INIT This function computes the initial values of the centroids
%   for k-means algorithm, depending on the chosen method.
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k     : (double), chosen k clusters
%       o init  : (string), type of initialization {'random','uniform'}
%
%   output ----------------------------------------------------------------
%
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N                   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Mu = zeros(N, k);

switch init
	case 'random'
		Mu = datasample(X,k,2,'Replace',false);
	case 'uniform'
		min_x = min(X,[],2);
		max_x = max(X,[],2);
		Mu = (max_x - min_x) .* rand(N,k) + min_x;
	otherwise
		error('Unkown init method');
end

end