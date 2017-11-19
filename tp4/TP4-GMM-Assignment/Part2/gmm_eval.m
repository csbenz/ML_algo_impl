function [AIC_curve, BIC_curve] =  gmm_eval(X, K_range, repeats, cov_type)
%GMM_EVAL Implementation of the GMM Model Fitting with AIC/BIC metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%       o AIC_curve  : (1 X K), vector of max AIC values for K-range
%       o BIC_curve  : (1 X K), vector of max BIC values for K-range
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = numel(K_range);
AIC_curve = zeros(1,K);
BIC_curve = zeros(1,K);

[Priors0, Mu0, ~, Sigma0] = my_gmmInit(X, K, cov_type);

for k=1:K
    aic_max = intmin;
    bic_max = intmin;
    for i=1:repeats
        [Priors, Mu, Sigma, ~] = my_gmmEM(X, K, cov_type,  Priors0, Mu0, Sigma0, 500);
        [aic_current, bic_current] = gmm_metrics(X, Priors, Mu, Sigma, cov_type);

        if(aic_current > aic_max || bic_current > bic_max)
            AIC_curve(k) = aic_current;
            BIC_curve(k) = bic_current;
            aic_max = aic_current;
            bic_max = bic_current;
        end
    end
end

AIC_curve
BIC_curve

end