%%  Test Implementation of Principal Component Analysis (PCA)
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Load 2D PCA Testing Dataset                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

% Dataset Path
dataset_path = '/home/christo/Documents/epfl-master4/ml_prog/tp1/TP1-PCA-Dataset/';

% Load 2D Testing Dataset for PCA
load(strcat(dataset_path,'2D_Gaussian.mat'))

% Visualize Dataset
options.labels      = labels;
options.title       = 'X = 2D Random Gaussian';

h0 = ml_plot_data(X',options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                Task 1: my_pca.m function                   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);

% Test my_pca.m against ML_toolbox numerically
try
    test_mypca(X, V, L, Mu);
catch
    error('Something is wrong with your output!')
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  Task 2: project_pca.m                     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Project Data to Choosen Principal Components
p = 2;
[A_p, Y] = project_pca(X, Mu, V, p );

% Test project_pca.m against ML_toolbox numerically
try
    test_projectpca(A_p, Y, X, Mu, V, p);
catch
    error('Something is wrong with your output!')
end

% Visualize Projected Data
plot_options             = [];
plot_options.is_eig      = true;
plot_options.labels      = labels;
plot_options.class_names = '2D Gauss';
plot_options.title       = 'My Projected data PCA';
h1 = ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Test reconstruct_pca.m                   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Project Data to 1d
p = 1;
[A_p, Y] = project_pca(X, Mu, V, p);

% Reconstruct Lossy Data from 1d
[X_hat]  = reconstruct_pca(Y, A_p, Mu);

% Test reconstruct_pca.m against ML_toolbox numerically
try
    test_reconstructpca(X_hat, Y, A_p, Mu);
catch
    error('Something is wrong with your output!')
end

%% Estimate Reconstruction Error
[e_rec]  = reconstruction_error(X, X_hat);
fprintf('Reconstruction Error with p=%d is %f \n',p,e_rec);

% Test reconstruct_error.m against ML_toolbox numerically
try
    test_reconstructionerror(e_rec, X, X_hat);
catch
    error('Something is wrong with your output!')
end

% Visualize Reconstructed Data
options.labels      = labels;
options.title       = 'Xhat : Reconstructed Data with p=1';
h0 = ml_plot_data(X_hat',options);
axis equal;
