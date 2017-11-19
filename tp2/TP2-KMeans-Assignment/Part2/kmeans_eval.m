function [RSS_curve, AIC_curve, BIC_curve] =  kmeans_eval(X, K_range,  repeats, init, type, MaxIter)
%KMEANS_EVAL Implementation of the k-means evaluation with clustering
%metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o init     : (string), type of initialization {'random','uniform','plus'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%
%   output ----------------------------------------------------------------
%       o RSS_curve  : (1 X K_range), RSS values for each value of K \in K_range
%       o AIC_curve  : (1 X K_range), AIC values for each value of K \in K_range
%       o BIC_curve  : (1 X K_range), BIC values for each value of K \in K_range
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RSS_curve = zeros(1, length(K_range));
AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));

iter = 0;

for i=1:length(K_range)
	k = K_range(i);

	RSS = zeros(1,repeats);
	AIC = zeros(1,repeats);
	BIC = zeros(1,repeats);

	for j=1:repeats
		 [labels,Mu,~,~] = my_kmeans(X,k,init,type,MaxIter,iter);
		[RSS(j),AIC(j),BIC(j)] = my_metrics(X, labels, Mu);
	end

	RSS_curve(i) = mean(RSS);
	AIC_curve(i) = mean(AIC);
	BIC_curve(i) = mean(BIC);
end


end