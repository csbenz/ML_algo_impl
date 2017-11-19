function [TP_rate_F_fold, FP_rate_F_fold, std_TP_rate_F_fold, std_FP_rate_F_fold] =  cross_validation(X, y, F_fold, tt_ratio, k_range)
%CROSS_VALIDATION Implementation of F-fold cross-validation for kNN algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o tt_ratio  : (double), Training/Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%       o TP_rate_F_fold  : (1 x K), True Positive Rate computed for each value of k averaged over the number of folds.
%       o FP_rate_F_fold  : (1 x K), False Positive Rate computed for each value of k averaged over the number of folds.
%       o std_TP_rate_F_fold  : (1 x K), Standard Deviation of True Positive Rate computed for each value of k.
%       o std_FP_rate_F_fold  : (1 x K), Standard Deviation of False Positive Rate computed for each value of k.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = length(k_range);

TP_rate = zeros(F_fold, K);
FP_rate = zeros(F_fold, K);

for f=1:F_fold 
    [X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio);
    [TP_rate(f,:), FP_rate(f,:)] = knn_ROC(X_train, y_train, X_test, y_test, k_range);
end

TP_rate_F_fold = mean(TP_rate);
FP_rate_F_fold = mean(FP_rate);
std_TP_rate_F_fold = std(TP_rate);
std_FP_rate_F_fold = std(FP_rate);

end
