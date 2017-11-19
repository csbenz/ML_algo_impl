function [ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio )
%SPLIT_DATA Randomly partitions a dataset into train/test sets using
%   according to the given tt_ratio
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y        : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o tt_ratio : train/test ratio.
%   output ----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, M] = size(X);
    
    % compute train and test sizes
    M_train = ceil(tt_ratio * M);
    
    P = randperm(M);
    P_train = P(1:M_train);
    P_test = P(M_train+1:M);
    
    X_train = X(:,P_train);
    X_test = X(:,P_test);
    
    y_train = y(P_train);
    y_test = y(P_test);
end

