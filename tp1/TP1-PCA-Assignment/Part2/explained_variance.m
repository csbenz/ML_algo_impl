function [ exp_var, cum_var, p ] = explained_variance( L, Var )
%EXPLAINED_VARIANCE Function that returns the optimal p given a desired
%   explained variance. The student should convert the Eigenvalue matrix 
%   to a vector and visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%       o Var    : (1 x 1), Desired Variance to be explained
%  
%   output ----------------------------------------------------------------
%
%       o exp_var  : (N x 1) vector of explained variance
%       o cum_var  : (N x 1) vector of cumulative explained variance
%       o p        : optimal principal components given desired Var


% ====================== Implement Eq. 8 Here ====================== 
L = diag(L);
sum_l = sum(L);
expl_var = L ./ sum_l;

% ====================== Implement Eq. 9 Here ====================== 
cum_var = cumsum(expl_var);

% ====================== Implement Eq. 10 Here ====================== 
p = 1;

while (p < size(L,1))
    if (cum_var(p) > Var)
        break;
    end
    
    p = p + 1;
end

% Visualize/Plot Explained Variance from Eigenvalues
figure;
plot(cum_var, '--r', 'LineWidth', 2);
hold on;

plot(p, cum_var(p), 'or')
grid on

end

