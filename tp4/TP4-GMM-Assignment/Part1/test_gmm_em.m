%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test Implementation of EM for GMM Algorithm on 2D Datasets. %%
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
%%         1) Load 2D GMM-EM Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (1a) Load 2D data for testing Gausspdf and Covariance Matrices
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
load(strcat(dataset_path,'/2D-Gaussian.mat'));
% Visualize Dataset
options.class_names = {};
options.title       = '2D Gaussian Dataset';
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
dataset = '1a';

%% (1b) Load 2D dataset for testing GMM-EM & Likelihood
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
load(strcat(dataset_path,'/2D-GMM.mat'));
% Visualize Dataset
options.labels      = labels;
options.class_names = {};
options.title       = '2D GMM Dataset';
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors,0.5);
grid on; box on;
dataset = '1b';

%% 1c) Draw Data with ML_toolbox GUI
close all; clc;
if exist('X'); clear X; end
if exist('labels'); clear labels;end
[X, labels] = ml_draw_data();
dataset = '1d';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Test my_gaussPDF.m function                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1a) before running this code block! %%%%%%
switch dataset
    case '1a' % Real Mu and Sigma used for 1a
        Mu = [1;1];
        Sigma = [1, 0.5; 0.5, 1];
    otherwise
        error('This testing function only works with datasets 1a!')        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_gaussPDF.m implementation
pts = test_mygaussPDF(X, Mu, Sigma);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Test my_gmmloglik.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1a or 1b) before running this code block! %%%%%%
switch dataset
    case '1a' % Real Mu and Sigma used for 1a
        Mu = [1;1];
        Sigma = [1, 0.5; 0.5, 1];
        Priors = [1];
    case '1b' % Load gmm parameters with real values        
        Priors = gmm.Priors;
        Mu     = gmm.Mu;
        Sigma  = gmm.Sigma;
    otherwise
        error('This testing function only works with datasets 1a and 1b!')        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_gmmLogLik.m implementation
pts = test_mygmmLogLik(X, Priors, Mu, Sigma);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            4) Test my_covariance.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_covariance.m implementation
pts = test_mycovariance(X); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Visualize different covariance matrices %%%%
[Sigma_full, Sigma_diag, Sigma_iso] = visualize_covariances(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          5) Test my_gmmInit.m function         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_gmmInit.m implementation
pts = test_mygmmInit(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set GMM Hyper-parameters
K = 2; cov_type = 'full';

% Run GMM-INIT function, estimates and visualizes initial parameters for EM algorithm
[ Priors0, Mu0, labels0, Sigma0 ] = my_gmmInit(X, K, cov_type);

%%%%%% Visualize Initial Estimates %%%%%%
options.labels      = labels0;
options.class_names = [];
options.title       = 'Initial Estimates for EM-GMM';
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu0',colors);hold on;
plot_gmm_contour(gca,Priors0,Mu0,Sigma0,colors);
grid on; box on;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           6) Test my_gmmEM.m function          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set GMM Hyper-parameters
K = 6; cov_type = 'iso';  plot_iter = 1; Max_iter = 500;
[Priors0, Mu0, ~, Sigma0] = my_gmmInit(X, K, cov_type);

%%%%%% Visualize Initial Estimates %%%%%%
options.labels      = [];
options.class_names = [];
options.title       = 'Initial Estimates for EM-GMM';
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu0',colors);hold on;
plot_gmm_contour(gca,Priors0,Mu0,Sigma0,colors);
grid on; box on;

%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_gmmEM.m implementation
pts = test_mygmmEM(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
