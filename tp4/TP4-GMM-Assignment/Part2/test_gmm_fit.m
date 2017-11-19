%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            Test GMM Model Fitting on 2D Datasets.           %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FILL IN PATHS TO PRACTICAL, DATASET & ML_TOOLBOX
clear all; close all; clc;
% Practical Path** <-- Fill in this path
tp_path = '/home/christo/Documents/epfl-master4/ml_prog/tp4/TP4-GMM-Assignment/';

addpath(genpath(tp_path))

% ML_Toolbox Path** <-- Fill in this path
mltoolbox_path = '/home/christo/Documents/epfl-master4/ml_prog/ML_toolbox/';
addpath(genpath(mltoolbox_path))

% Dataset Path** <-- Fill in this path
dataset_path = '/home/christo/Documents/epfl-master4/ml_prog/tp4/TP4-GMM-Datasets/';



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D GMM Fit Function Testing Dataset        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1a) Load 2d GMM Dataset
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
load(strcat(dataset_path,'/2D-GMM.mat'))

% Visualize Dataset
options.labels      = labels;
options.class_names = {};
options.title       = '2D GMM Dataset';
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors);
grid on; box on;

%% 1b) Load 2d Circle Dataset
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
load(strcat(dataset_path,'/2d-concentric-circles.mat'))

% Visualize Dataset
options.class_names = {};
options.title       = '2D Concentric Circles Dataset';
options.labels       = y;
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;


%% 1c) Draw Data with ML_toolbox GUI
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
[X, labels] = ml_draw_data();
dataset = '1d';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             2) Check gmm_metrics.m function                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test gmm_metrics.m implementation
pts = test_gmmMetrics(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              3) Choosing K test gmm_eval.m                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test gmm_eval.m implementation
pts = test_gmmeval(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% K-means Evaluation Parameters
K_range=[1:10]; cov_type = 'full'; repeats = 10;

% Evaluate gmm-em to find the optimal k
[AIC_curve, BIC_curve] = gmm_eval(X, K_range, repeats, cov_type);

% Plot Metric Curves
figure('Color',[1 1 1]);
plot(AIC_curve,'--o', 'LineWidth', 1); hold on;
plot(BIC_curve,'--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('AIC', 'BIC')
title(sprintf('GMM (%s) Model Fitting Evaluation metrics',cov_type))
grid on

%% Pick best K from Plot and Visualize result
% Set GMM Hyper-parameters 
K = 3; cov_type = 'full'; %<== CHANGE VALUES HERE!
Max_iter = 500;
%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
[Priors0, Mu0, ~, Sigma0] = my_gmmInit(X, K, cov_type);
[Priors, Mu, Sigma, iter] = my_gmmEM(X, K, cov_type, Priors0, Mu0, Sigma0, Max_iter);
toc;

%%%%%% Visualize Final Estimates %%%%%%
options.labels      = [];
options.class_names = {};
options.plot_figure = false;
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu',colors);hold on;
plot_gmm_contour(gca,Priors,Mu,Sigma,colors);
title(sprintf('Final GMM Parameters iter= %d',iter));
grid on; box on;

% Visualize GMM pdf from learnt parameters
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)
