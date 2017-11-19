function [F1_overall, P, R, F1] =  my_f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (M x 1),  M-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels   : (M x 1),  M-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o F1_overall      : (1 x 1)     f1-measure for the clustered labels
%       o P               : (nClusters x nClasses)  Precision values
%       o R               : (nClusters x nClasses)  Recall values
%       o F1              : (nClusters x nClasses)  F1 values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
M         = length(class_labels);
true_K    = unique(class_labels);
found_K   = unique(cluster_labels);
nClasses  = length(true_K);
nClusters = length(found_K);

% Output Variables
P = zeros(nClusters, nClasses);
R = zeros(nClusters, nClasses);
F1 = zeros(nClusters, nClasses);
F1_overall = 0;


for i=1:nClusters
    for j=1:nClasses

        cluster_c = 0;
        class_c = 0;
        cluster_and_class_c = 0;
        
        for m=1:M
            if cluster_labels(m) == i
                cluster_c = cluster_c + 1;
            end
            
            if class_labels(m) == j
                class_c = class_c + 1;
            end
            
            if(cluster_labels(m) == i && class_labels(m) == j)
                cluster_and_class_c = cluster_and_class_c + 1;
            end
        end
        
        if class_c ~= 0
            P(i,j) = cluster_and_class_c / cluster_c;
        end
        
        if cluster_c ~= 0
            R(i,j) = cluster_and_class_c / class_c;
        end
        
        if (R(i,j)+P(i,j)) ~= 0
            F1(i,j) = (2 * R(i,j) * P(i,j)) / (R(i,j) + P(i,j));
        end
    end
end

for i=1:nClasses
    f1_max = max(F1(:,i));
    
    count = 0;
    for m=1:M
        if class_labels(m) == i
            count = count + 1;
        end
    end
    
    F1_overall = F1_overall + ((count / M) * f1_max);
end



end
