function [ ll ] = my_gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o ll       : (1 x 1) , loglikelihood
%%


[~,M] = size(X);
K = length(Priors);

pdfs = zeros(K,M);
for k=1:K
    pdf = my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
    pdfs(k,:) = pdf;
end

ll = 0;
for i=1:M
    p = 0;
    for k=1:K
        p = p + Priors(k) * pdfs(k,i);
    end
    ll = ll + log(p);
end


end

