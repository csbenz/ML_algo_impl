%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Test Implementation of K-NN Algorithm on 2D Datasets. %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FILL IN PATHS TO PRACTICAL, DATASET & ML_TOOLBOX
clear all; close all; clc;
% Practical Path** <-- Fill in this path
tp_path = '/home/christo/Documents/epfl-master4/ml_prog/tp3/TP3-KNN-Assignment/';

addpath(genpath(tp_path))

% ML_Toolbox Path** <-- Fill in this path
mltoolbox_path = '/home/christo/Documents/epfl-master4/ml_prog/ML_toolbox/';

addpath(genpath(mltoolbox_path))

% Dataset Path** <-- Fill in this path
dataset_path = '/home/christo/Documents/epfl-master4/ml_prog/tp3/TP3-KNN-Datasets/';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           1) Load 2D KNN Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load Concentric Circle Data
% Load Dataset
load(strcat(dataset_path,'/2d-concentric-circles.mat'))
[~, M] = size(X);
rand_idx = randperm(M);
X = X(:,rand_idx);
y = y(rand_idx);

% Visualize Dataset
options.labels      = y;
options.class_names = {'y = 0','y = 1'};
options.title       = '2D Concentric Circles Dataset';

h0 = ml_plot_data(X',options);
axis equal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     2) Data Handling for Classification (split_data.m)        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Training/Testing Ratio
tt_ratio = 0.3; % 30% of points = train / 70% of points = testing

% Split data into a training dataset that kNN can use to make predictions 
% and a test dataset that we can use to evaluate the accuracy of the model.
[X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test split_data.m function for X and a tt_ratio defined above
pts = test_datasplits(X, tt_ratio, X_train, X_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Split t
options.labels      = y_train;
options.class_names = [];
options.title       = sprintf('Data Split for 2D Dataset tt-ratio: %1.2f',tt_ratio);
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X_train',options); hold on;
scatter(X_test(1,:),X_test(2,:),50,'^','MarkerFaceColor',[1 1 0.5],'MarkerEdgeColor', [0 0 0]);
legend({'$y=0$','$y = 1$','$\mathbf{x}\prime$'},'Interpreter','latex')
axis equal


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Test kNN implementation (my_knn) and Visualize Results  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select k
k = 5; d_type = 'L2';

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, k, d_type);

% Visualize Split Dataset
options.labels      = y;
options.class_names = [];
options.title       = sprintf('My kNN, tt-ratio = %1.2f, k= %d',tt_ratio, k);
if exist('h1','var') && isvalid(h1), delete(h1);end
h2 = ml_plot_data(X',options); hold on;
scatter(X_test(1,y_est == 0),X_test(2,y_est == 0),150,'o','MarkerEdgeColor', [1 0 0]);hold on;
scatter(X_test(1,y_est == 1),X_test(2,y_est == 1),150,'o','MarkerEdgeColor', [0 0 1]);
legend({'$y=0$','$y = 1$','$\hat{y} = 0$','$\hat{y} = 1$'},'Interpreter','latex')
axis equal
pause(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_knn.m function for X and a tt_ratio defined above
pts =  test_myknn(X_train, y_train, X_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             4) Test my_accuracy.m function                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select k
k = 7; 

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, k, 'L2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test my_accuracy.m function for your estimated labels
pts = test_myaccuracy(y_test, y_est);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           5) Visualize kNN Results and Accuracy            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select k
k = 1; d_type = 'L2';

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, k, d_type);

% Compute Accuracy
acc =  my_accuracy(y_test, y_est);

% Visualize Split Dataset
options.labels      = y;
options.class_names = [];
options.title       = sprintf('My kNN, tt-ratio = %1.2f, k= %d, Acc = %1.3f',tt_ratio, k, acc);
h2 = ml_plot_data(X',options); hold on;
scatter(X_test(1,y_est == 0),X_test(2,y_est == 0),150,'o','MarkerEdgeColor', [1 0 0]);hold on;
scatter(X_test(1,y_est == 1),X_test(2,y_est == 1),150,'o','MarkerEdgeColor', [0 0 1]);
legend({'$y=0$','$y = 1$','$\hat{y} = 0$','$\hat{y} = 1$'},'Interpreter','latex')
axis equal

% Plot K-NN Decision boundary
knn_options.k      = k;
knn_options.d_type = d_type;
[~, model]= knn_classifier(X_train, y_train, [], knn_options);
f_knn     = @(X_test)knn_classifier(X_test, [], model, []);

% Plot Decision Boundary
clc;
c_options         = [];
plot_data_options = [];
c_options.dim_swaped     = true;
c_options.show_misclass  = false;
c_options.title          = sprintf('K(%d)-NN Decision Boundary with TT/ratio: %1.2f',k,tt_ratio);
if exist('hc','var') && isvalid(hc), delete(hc);end
hc = ml_plot_classifier(f_knn,X',y,c_options,plot_data_options);
axis tight

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         6) Choosing K by visualizing knn_eval.m            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Range of K to test accuracy
M_train = length(X_train);
k_range = [1:2:ceil(M_train/2)];
acc_curve = knn_eval(X_train, y_train, X_test, y_test, k_range); 

% Plot Accuracy Curve
if exist('h_acc','var')     && isvalid(h_acc),     delete(h_acc);    end
h_acc = figure;hold on;
plot(k_range,acc_curve,'--o', 'LineWidth', 1, 'Color', [1 0 0]); hold on;
xlabel('k'); ylabel('Acc')
title('Classification Evaluation for KNN')
grid on
pause(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test knn_eval.m function for your estimated labels
pts = test_knneval(X_train, y_train, X_test, y_test, k_range);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%