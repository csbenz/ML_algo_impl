%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          Test Implementation of K-Means Metrics            %%
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
%% 1a) Load 2D data sampled from a GMM
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

%% 1b) Load 2d Ripley Dataset
% Load Dataset
load(strcat(dataset_path,'/2d-ripley.mat'))

options.class_names = {};
options.labels      = labels;
options.title       = '2D Ripley Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Test my_metrics.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run k-means Once for metric evaluation
clc; close all;
K = 6; init='random'; type='L2'; MaxIter = 100; plot_iter = 0;
[labels, Mu] =  my_kmeans(X, K, init, type, MaxIter, plot_iter);

% Plot decision boundary
my_kmeans_result.distance    = type;
my_kmeans_result.K           = K;
my_kmeans_result.method_name = 'kmeans';
my_kmeans_result.labels      = labels';
my_kmeans_result.centroids   = Mu';
my_kmeans_result.title       = sprintf('. My K-means result. K = %d, dist = %s',K, type);
if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_class_boundary(X',my_kmeans_result);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_metrics.m function
% on Mu, labels from my_kmeans
test_mymetrics(X, Mu, labels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Choosing K test kmeans_eval.m               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10; init = 'random'; MaxIter = 100;

% Evaluate K-means to find the optimal K
[RSS_curve,AIC_curve,BIC_curve] = kmeans_eval(X, K_range, repeats, init, type, MaxIter);

% Plot Metric Curves
if exist('h_metrics','var') && isvalid(h_metrics),  delete(h_metrics); end
h_metrics = figure('Color',[1 1 1]);hold on;
plot(RSS_curve,'--o', 'LineWidth', 1); hold on;
plot(AIC_curve,'--o', 'LineWidth', 1); hold on;
plot(BIC_curve,'--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('RSS', 'AIC', 'BIC')
title('Clustering Evaluation metrics')
grid on
axis tight

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test kmeans_eval.m function with previously defined parameters
test_kmeanseval(X, K_range, repeats, init, type, MaxIter);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
