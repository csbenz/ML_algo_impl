%%  Test Implementation of Principal Component Analysis (PCA)
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
addpath(genpath('ML_toolbox'))
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    Load YALE Face Dataset                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

% Dataset Path
dataset_path = './TP1-PCA-Dataset/Faces/';

% Load 2D Testing Dataset for PCA
load(strcat(dataset_path,'Yale_32x32.mat'))

% Generate Variables
X       = fea;
labels  = gnd';
[M, N]  = size(X);
sizeIm  = sqrt(N);

% Plot 64 random samples of the dataset
idx = randperm(size(X,1));
h0  = ml_plot_images(X(idx(1:64),:),[sizeIm sizeIm]);

% Transpose for PCA
X = fea';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Task 7: Test your plot_eigenfaces.m function        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function plots the first 20 eigenfaces and returns the 
% projections variabls from my_pca and the eigenfaces in a 3-d array of
% size: sizeIm x sizeIm x 20

[V, L, Mu, eigenfaces] = plot_eigenfaces(X, sizeIm);

% Test pca_eigenfaces.m against ML_toolbox numerically
try
    test_pcaeigenfaces(V, eigenfaces , sizeIm);
catch
    error('Something is wrong with your output!')
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Task 8: Test your reconstruction_eigenfaces.m function   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function reconstructs the first face image from the dataset with
% p = 1, p = 51, p = 101, p = 151, it should display these images +
% the mean image of the dataseta and the original first image
[reconstructed_faces] = reconstruction_eigenfaces(X, V, Mu, sizeIm);

% Test reconstruction_eigenfaces.m against ML_toolbox numerically
try
    test_reconstructeigenfaces(reconstructed_faces, X, V, Mu, sizeIm);
catch
    error('Something is wrong with your output!')
end

