%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Test Implementation of K-Means Algorithm on 2D Datasets. %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FILL IN PATHS TO PRACTICAL, DATASET & ML_TOOLBOX
clear all; close all; clc;
% Practical Path** <-- Fill in this path
tp_path = '/home/christo/Documents/epfl-master4/ml_prog/tp2/TP2-KMEANS-Assignment/';
addpath(genpath(tp_path))

% ML_Toolbox Path** <-- Fill in this path
mltoolbox_path = '/home/christo/Documents/epfl-master4/ml_prog/tp2/ML_toolbox/';
addpath(genpath(mltoolbox_path))

% Dataset Path** <-- Fill in this path
dataset_path = '/home/christo/Documents/epfl-master4/ml_prog/tp2/TP2-KMEANS-Datasets/';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D KMEAN Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1a) Load 2D data sampled from a GMM

% Load Dataset
load(strcat(dataset_path,'/2d-gmm-4.mat'))

% Visualize Dataset
options.class_names = {};
options.title       = '2D Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);
ml_plot_sigma (gmm, colors, 10);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Test my_distance.m function                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_distance.m function on random sample from X
test_mydistance(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Test kmeans_init.m function                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default Init Values to Check kmeans_init.m
K = 4; init = 'random'; % init: {'random','uniform'}
Mu =  kmeans_init(X, K, init);

% Visualize Initialized Centroids
if exist('h1','var') && isvalid(h1), delete(h1);end
options.title       = sprintf('Centroid Initialization: %s', init);
h1 = ml_plot_data(X',options); hold on;
colors     = hsv(K);
ml_plot_centroid(Mu',colors);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test kmeans_init.m function for both 'random' and 'uniform'
test_kmeansinit(X,K);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             4) Test my_distX2Mu.m function                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default Init Values to Test my_distX2Mu.m
K = 4; init = 'random';
Mu =  kmeans_init(X, K, init);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_distX2Mu.m function
test_mydistX2Mu(X, Mu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              5) Test my_kmeans.m function                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Init Values to Check my_kmeans.m - you should play around with these
K = 4; init='random'; type='L2'; MaxIter = 100; plot_iter = 1; clc; close all;
[labels, Mu, ~] =  my_kmeans(X, K, init, type, MaxIter, plot_iter);
pause(1);

% Plot decision boundary
my_kmeans_result.distance    = type;
my_kmeans_result.K           = K;
my_kmeans_result.method_name = 'kmeans';
my_kmeans_result.labels      = labels';
my_kmeans_result.centroids   = Mu';
my_kmeans_result.title       = sprintf('. My K-means result. K = %d, dist = %s',K, type);
if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_class_boundary(X',my_kmeans_result);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause(1);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_kmeans.m function for a range of K's
clc; K_range = [1:2:8];
for i=1:length(K_range)
    test_mykmeans(X, K_range(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
