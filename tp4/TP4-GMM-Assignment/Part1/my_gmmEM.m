function [  Priors, Mu, Sigma, iter ] = my_gmmEM(X, K, cov_type, Priors0, Mu0, Sigma0, Max_iter)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o Priors0   : (1 x K), the set of INITIAL priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu0       : (N x K), an NxK matrix corresponding to the INITIAL centroids 
%                           mu^(0) = {mu^1,...mu^K}
%       o Sigma0    : (N x N x K), an NxNxK matrix corresponding to the
%                    INITIAL Covariance matrices  Sigma^(0) = {Sigma^1,...,Sigma^K}
%       o Max_iter  : (1 x 1) maximum number of allowable iterations
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of FINAL priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the FINAL centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   FINAL Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o iter      : (1 x 1) number of iterations it took to converge
%%

[N,M] = size(X);
Priors = Priors0;
Mu = Mu0;
Sigma = Sigma0;
iter = 0;

ll = my_gmmLogLik(X, Priors, Mu, Sigma);

while true
    % Expectation
    prob = zeros(K,M);
    for k=1:K
        prob(k,:) = my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
    end

    post_prob = zeros(K,M);
    for i=1:M
        for k=1:K
            post_prob(k,i) = Priors(k) * prob(k,i) / sum(Priors * prob(:,i));
        end
    end
    
    % Maximization
    for k=1:K
        Priors(k) = sum(post_prob(k,:)) ./ M; 
        
        p_top = zeros(N,1);
        for i=1:M
            p_top = p_top + post_prob(k,i) * X(:,i); % TODO simplify bsxfun
        end
        Mu(:,k) = p_top / sum(post_prob(k,:));
        
        switch cov_type
            case {'full','diag'}
                p_top = zeros(N,N);
                for i=1:M
                    p_top = p_top + post_prob(k,i) *(X(:,i) - Mu(:,k)) * (X(:,i) - Mu(:,k)).';
                end
                Sigma(:,:,k) = p_top / sum(post_prob(k,:));
                
                if(strcmp(cov_type, 'diag'))
                    Sigma(:,:,k) = eye(N) .* diag(Sigma(:,:,k));
                end
            case 'iso'
                p_top = 0;
                for i=1:M
                    p_top = p_top + post_prob(k,i) * norm(X(:,i) - Mu(:,k))^2;
                end 
                Sigma(:,:,k) = diag(ones(1,N) * p_top) / (N * sum(post_prob(k,:)));
        end
        
        % variance to avoid singular matrice
        Sigma(:,:,k) = Sigma(:,:,k) + (eye(N) * 1e-5);
    end
    
    % loop
    iter = iter + 1;
    
    % Stop condition
    old_ll = ll;
    ll = my_gmmLogLik(X, Priors, Mu, Sigma);
    if(iter >= Max_iter || abs(old_ll - ll) <= 1e-3)
        break;
    end
    
end

end

