function [labels, Mu, Mu_init, iter] =  my_kmeans(X, K, init, type, MaxIter, plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'random','uniform'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%       o Mu_init  : (N x k), same as above, corresponds to the centroids used
%                            to initialize the algorithm
%       o iter     : (int), iteration where algorithm stopped
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);
if plot_iter == [];plot_iter = 0;end

% Output Variables
labels  = zeros(1,M);
Mu      = zeros(N, K);
Mu_init = zeros(N, K);
iter      = 0;

% Step 1. Mu Initialization
Mu_init = kmeans_init(X,K,init);
Mu = Mu_init;

%%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%
% Visualize Initial Centroids if N=2 and plot_iter active
colors     = hsv(K);
if (N==2 && plot_iter)
    options.title       = sprintf('Initial Mu with <%s> method', init);
    ml_plot_data(X',options); hold on;
    ml_plot_centroid(Mu_init',colors);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


iter = 0;
while true

    %%%%% Implement K-Means Algorithm HERE %%%%%    
    %Mu_previous = Mu;
    
    % Step 2. Distances from X to Mu
	d = my_distX2Mu(X,Mu,type);
	% (k,M)
    
	% Step 3. Assignment Step: Mu Responsability
	% Equation 5 and 6
	%[~, k_i] = min(d,[],2); 
    for i=1:K
        for j=1:M
            if d(i,j) == min(d(:,j))
                labels(j) = i;
            end
        end
    end

	r_i = zeros(K,M);
	for i=1:M
		r_i(labels(i),i) = 1;
	end
	
	
	% Step 4. Update Step: Recompute Mu	
    % Equation 7
	for k=1:K
		%Mu(:,k) = sum(r_i(k,:).' @ X) / (sum(r_i(k,:)) %maybe r_i not transpose?
        if sum(r_i(k,:)) ~= 0
            Mu(:,k) = (r_i(k,:) * X.') ./ (sum(r_i(k,:)));
        end
    end
    
    

	% Check for stopping conditions (Mu stabilization or MaxIter)
    if abs(Mu-Mu_init) < 0.01
		%sprintf('Reached solution');
		break;
    end
    
    Mu_init = Mu;

	if (iter > MaxIter)
		%error('Reached max number of iter');
		break;
	end
    
    %%%%%%%%%%%%%%%%%         TEMPLATE CODE      %%%%%%%%%%%%%%%%       
    if (N==2 && iter == 1 && plot_iter)
        options.labels      = labels;
        options.title       = sprintf('Mu and labels after 1st iter');
        ml_plot_data(X',options); hold on;
        ml_plot_centroid(Mu',colors);
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    iter = iter+1;
    
    
    % Check if all clusters have been assigned
    if not(all(ismember(1:K,labels)))
        % sprintf('Some clusters are empty, recomputing...');
        Mu_init = kmeans_init(X,K,init);
        Mu = Mu_init;
    end
    
end


%%%%%%%%%%%   TEMPLATE CODE %%%%%%%%%%%%%%%
if (N==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroid(Mu',colors);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end